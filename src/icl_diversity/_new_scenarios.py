"""Scenario 7 & 8 mode definitions for ICL diversity metric validation.

Scenario 7: Open Creative — 15 format/genre modes, 100 responses per prompt
Scenario 8: Problem-Solving — 15 approach modes, 100 responses per prompt

These are separated from the main scenarios.py to avoid Python 3.10 f-string
backslash restrictions in the original file.
"""

import random
from typing import Callable


def _generate_high_diversity_responses(
    modes: list[Callable[[random.Random], str]],
    n: int = 100,
    seed: int = 0,
) -> list[str]:
    rng = random.Random(seed)
    responses: list[str] = []
    indices = list(range(n))
    rng.shuffle(indices)
    mode_assignments = [i % len(modes) for i in indices]
    paired = sorted(zip(indices, mode_assignments))
    for _, mode_idx in paired:
        responses.append(modes[mode_idx](rng))
    return responses


# ============================================================================
# Pre-defined choice tuples for modes that need special characters
# ============================================================================

# --- Scenario 7: Open Creative ---

# Rain modes
_RAIN_HAIKU_LINE2 = (
    "a river is born from sky",
    "the earth drinks in silence",
    "puddles mirror clouds above",
    "the world sighs and glistens",
)
_RAIN_CODE_CONTAINERS = ("[]", "list()", "collections.deque()", "np.array([])")
_RAIN_CODE_DROPS = (
    "random.gauss(0, 1)",
    "random.uniform(0, 10)",
    "math.sin(t) * random.random()",
    "poisson.rvs(3.2)",
)
_RAIN_DIARY_LOCATIONS = (
    "summer rain in Kyoto",
    "monsoon in Mumbai",
    "spring shower in London",
    "autumn downpour in Seattle",
)
_RAIN_LIST_ITEM1 = (
    "Rain is recycled water that has evaporated from oceans and lakes",
    "The smell of rain is called petrichor",
    "A single thunderstorm can drop 500 million gallons of water",
    "Rain falls on every continent including Antarctica",
)
_RAIN_LIST_ITEM2 = (
    "Mawsynram, India receives 467 inches of rain per year",
    "The wettest place on Earth gets over 11 meters of rain annually",
    "Some deserts receive less than 1 inch of rain per year",
    "Antarctica is technically the driest continent",
)
_RAIN_LIST_ITEM3 = (
    "Raindrops are not teardrop-shaped; they are flattened spheres",
    "Raindrops can fall at up to 22 mph",
    "It takes about 10 minutes for a raindrop to reach the ground",
    "The largest raindrops ever recorded were 8.8 mm across",
)
_RAIN_LIST_ITEM4 = (
    "Acid rain was first identified in 1852",
    "Rain on Venus is made of sulfuric acid",
    "On Titan, it rains liquid methane",
    "Diamond rain may fall on Neptune",
)
_RAIN_LIST_ITEM5 = (
    "Phantom rain evaporates before hitting the ground",
    "Red rain fell in Kerala, India in 2001",
    "Fish have been reported falling with rain",
    "Rain contains dissolved nitrogen that fertilizes soil",
)
_RAIN_DIALOGUE_QUESTIONS = (
    "Do you think it will rain?",
    "Looks like rain again.",
    "Did you bring an umbrella?",
    "Can you hear that thunder?",
)
_RAIN_DIALOGUE_ANSWERS = (
    "The forecast says so.",
    "I hope not, I left my coat.",
    "Certainly smells like it.",
    "I love the sound of it.",
)
_RAIN_PHILOSOPHY = (
    "falls on the rich and poor alike",
    "does not discriminate between palace and gutter",
    "treats every surface the same",
    "makes no distinction between king and beggar",
)
_RAIN_PHILOSOPHY2 = (
    "There is something humbling about standing in a downpour",
    "We are reminded of our smallness when the sky opens",
    "In rain, we see the cycle that sustains all life",
    "Each drop is both an ending and a beginning",
)
_RAIN_SONG_LINE2 = (
    "washing all the dust from this old town",
    "drumming on the rooftops of the town",
    "turning every field to shining brown",
    "soaking through my coat and evening gown",
)
_RAIN_SONG_LINE3 = (
    "I stand beneath the silver sky and sing",
    "The gutters overflow and church bells ring",
    "The children splash in puddles everything",
    "The flowers bow their heads accepting spring",
)
_RAIN_METAPHOR_SUBJECTS = ("drum", "canvas", "mirror", "sleeping animal")
_RAIN_METAPHOR_VERBS = ("drummer", "painter", "truth", "whisper that woke it")
_RAIN_HIST = (
    "1931, the Yangtze River floods killed an estimated 3.7 million people",
    "1887, the Yellow River flood became one of the deadliest natural disasters in history",
    "2005, Hurricane Katrina brought record rainfall to the Gulf Coast",
    "1970, the Bhola cyclone devastated East Pakistan with torrential rain",
)
_RAIN_HIST2 = (
    "Rainfall patterns have shaped the rise and fall of civilizations",
    "Ancient Egyptians depended entirely on the Nile floods for agriculture",
    "The Dust Bowl of the 1930s was caused partly by the absence of rain",
    "Monsoon failures triggered famines across South Asia for centuries",
)
_RAIN_LETTER_SIGNERS = ("Dr. A. Marsh", "Prof. L. Waters", "J. Pluvio", "R. Stormfield")
_RAIN_CHILD_NAMES = ("Nimbus", "Drizzle", "Splashy", "Puddles")
_RAIN_JSON_LOCATIONS = ("Seattle, WA", "London, UK", "Mumbai, IN", "Tokyo, JP")
_RAIN_JSON_DATES = ("2024-03-15", "2024-07-22", "2024-11-03", "2024-01-08")
_RAIN_JSON_TYPES = ("drizzle", "shower", "downpour", "thunderstorm")

# Seven modes
_SEVEN_HAIKU_LINE2 = (
    "counting dreams beneath the moon",
    "a number older than the hills",
    "luck spins on its axis now",
    "the world divided into parts",
)
_SEVEN_CODE_DOCS = (
    '"""Check if n relates to 7."""',
    "# Seven is considered lucky",
    "# Prime number check for 7",
    '"""The most magical digit."""',
)
_SEVEN_CODE_RETURNS = (
    "n % 7 == 0",
    "str(7) in str(n)",
    "n == 7 or n % 7 == 0",
    "sum(int(d) for d in str(n)) == 7",
)
_SEVEN_RECIPE_INGREDIENTS = (
    "cumin, coriander, turmeric, paprika, cinnamon, clove, cardamom",
    "beans, cheese, salsa, guacamole, sour cream, olives, lettuce",
    "wheat, oat, rye, barley, millet, flax, quinoa",
    "basil, parsley, chives, dill, tarragon, mint, cilantro",
)
_SEVEN_SCIENCE = (
    "there are 7 crystal systems in mineralogy",
    "nitrogen has atomic number 7",
    "light splits into 7 colors in a rainbow",
    "the pH scale centers around 7 as neutral",
)
_SEVEN_SCIENCE2 = (
    "It is the fourth prime number",
    "Seven is the most commonly chosen random number",
    "There are 7 SI base units",
    "A ladybug often has 7 spots",
)
_SEVEN_DIARY_EVENTS = (
    "I learned to ride a bicycle",
    "my family moved to a new city",
    "I found a four-leaf clover",
    "I read my first real book",
)
_SEVEN_DIARY_FOLLOW = (
    "That number has followed me ever since",
    "Seven has been my lucky number since then",
    "I still think of that year as magical",
    "Everything important happens in sevens for me",
)
_SEVEN_LIST_1 = (
    "7 continents on Earth",
    "7 colors in a rainbow",
    "7 notes in a musical scale",
    "7 days in a week",
)
_SEVEN_LIST_2 = (
    "7 wonders of the ancient world",
    "7 seas of classical antiquity",
    "7 hills of Rome",
    "7 liberal arts of medieval education",
)
_SEVEN_LIST_3 = (
    "7 deadly sins",
    "7 virtues in Christian theology",
    "7 sacraments",
    "7 heavens in Islamic tradition",
)
_SEVEN_LIST_4 = (
    "7 samurai in Kurosawa's film",
    "7 dwarfs in Snow White",
    "7 Harry Potter books",
    "7 Horcruxes in the series",
)
_SEVEN_LIST_5 = (
    "7 is the most common dice roll (2d6)",
    "007 is James Bond's number",
    "A 7-game series decides champions",
    "Lucky 7 in slot machines",
)
_SEVEN_LIST_6 = (
    "The Big Dipper has 7 stars",
    "7 chakras in Hindu tradition",
    "7 layers of the OSI model",
    "7 digits in a US phone number",
)
_SEVEN_LIST_7 = (
    "7 is a Mersenne prime (2^3-1)",
    "7 is a happy number",
    "1/7 = 0.142857... (6-digit repeating)",
    "7! = 5040, a highly composite factorial",
)
_SEVEN_DIALOGUE_Q = (
    "Why is seven so special?",
    "Pick a number, any number.",
    "What comes after six?",
    "Guess my favorite number.",
)
_SEVEN_DIALOGUE_A = (
    "Because seven ate nine!",
    "Seven. It is always seven.",
    "It is the only number that rhymes with heaven.",
    "Seven -- prime, indivisible, and perfect.",
)
_SEVEN_PHIL = (
    "perfect",
    "sacred",
    "mysterious",
    "fundamental",
)
_SEVEN_PHIL2 = (
    "It recurs in religion, music, science, and folklore",
    "Perhaps our minds are drawn to it because it sits at the edge of our working memory",
    "The ancients saw it everywhere: days, planets, notes",
    "It is large enough to feel complex yet small enough to hold in the mind",
)
_SEVEN_PHIL3 = (
    "Is seven special because we made it so, or did we discover its importance?",
    "Maybe seven is the universe whispering a pattern we almost understand",
    "The question is not why seven is special but why we need it to be",
    "Seven stands at the boundary between the countable and the mysterious",
)
_SEVEN_SONG_LINE2 = (
    "counting every blessing, every star",
    "a number written on the wind so far",
    "spinning like a wheel inside a jar",
    "playing on the strings of my guitar",
)
_SEVEN_SONG_LINE3 = (
    "Seven days, seven nights, seven reasons why",
    "Seven seas, seven skies, seven ways to fly",
    "Seven steps, seven breaths, seven times I try",
    "Seven doors, seven keys, seven questions high",
)
_SEVEN_MATH_PROPS = (
    "7 is prime, a Mersenne prime (2^3-1), and a safe prime",
    "7 is the smallest number that cannot be represented as a sum of fewer than 4 non-negative cubes",
    "1/7 has a 6-digit repeating decimal: 0.142857142857...",
    "7 is the fourth prime and the second Mersenne prime",
)
_SEVEN_METAPHOR1 = ("a prism", "a mirror", "a door", "a seed")
_SEVEN_METAPHOR2 = (
    "seven colors of the world emerge",
    "seven reflections stare back at you",
    "seven rooms open to the unknown",
    "seven branches grow toward light",
)
_SEVEN_HIST = (
    "Seven Wonders of the Ancient World were first listed by Antipater of Sidon around 140 BCE",
    "the Seven Years War (1756-1763) reshaped the global colonial order",
    "the Group of Seven (G7) was formed in 1975 to coordinate economic policy",
    "the seven liberal arts -- trivium and quadrivium -- defined medieval education",
)
_SEVEN_LETTER_PROPOSALS = (
    "propose a festival celebrating the number seven",
    "request that the seventh day be recognized as a civic holiday",
    "suggest naming seven streets after the seven wonders",
    "recommend a seven-part documentary series on the number",
)
_SEVEN_CHILD_QUESTS = (
    "find the seven magic stones",
    "count seven shooting stars",
    "make seven new friends",
    "visit seven enchanted places",
)
_SEVEN_JSON_CATEGORIES = ("mathematics", "culture", "nature", "history")
_SEVEN_JSON_FACTS = (
    "Fourth prime number",
    "Number of continents",
    "Days in a week",
    "Colors in a rainbow",
)

# Water modes
_WATER_HAIKU_LINE2 = (
    "still pools hold the sky within",
    "falling over ancient stone",
    "a river carries time away",
    "the ocean breathes and rests",
)
_WATER_CODE_DOCS = (
    '"""Simulate water flow."""',
    "# Model water cycle phases",
    '"""Calculate water volume."""',
    "# Track water state transitions",
)
_WATER_CODE_STATES = (
    "['liquid', 'solid', 'gas']",
    "['ice', 'water', 'steam']",
    "['frozen', 'flowing', 'evaporating']",
    "['condensed', 'liquid', 'vapor']",
)
_WATER_RECIPE_NAMES = (
    "Infused Water Medley",
    "Sparkling Water Refresher",
    "Herbal Water Tonic",
    "Citrus Water Blend",
)
_WATER_SCIENCE = (
    "Water is the only common substance found naturally in all three states on Earth",
    "A water molecule has a bond angle of 104.5 degrees",
    "Water has an unusually high specific heat capacity of 4.18 J/g K",
    "Pure water has a neutral pH of exactly 7.0 at 25 degrees Celsius",
)
_WATER_SCIENCE2 = (
    "Ice is less dense than liquid water, which is why it floats",
    "Water reaches maximum density at 4 degrees Celsius",
    "Surface tension allows insects to walk on water",
    "Water is sometimes called the universal solvent",
)
_WATER_DIARY_PLACES = (
    "swimming in a cold mountain lake",
    "watching waves crash on the shore in winter",
    "drinking from a spring in the Alps",
    "sitting by a stream in the forest after a long hike",
)
_WATER_DIARY_FEELINGS = (
    "I felt connected to something ancient and immense",
    "the simplicity of water made everything else fall away",
    "I understood why civilizations grew beside rivers",
    "nothing else mattered in that moment",
)
_WATER_LIST_ITEM1 = (
    "Water covers about 71% of the Earth's surface",
    "Only 2.5% of Earth's water is freshwater",
    "The human body is roughly 60% water",
    "A single tree can transpire 100 gallons of water per day",
)
_WATER_LIST_ITEM2 = (
    "The deepest point in the ocean is about 36,000 feet",
    "Lake Baikal contains 20% of the world's unfrozen freshwater",
    "Groundwater accounts for 30% of the world's freshwater",
    "About 96.5% of all water is in the oceans",
)
_WATER_LIST_ITEM3 = (
    "Water can dissolve more substances than any other liquid",
    "Hot water freezes faster than cold water (Mpemba effect)",
    "Water expands when it freezes, unlike most substances",
    "A water molecule is about 0.275 nanometers wide",
)
_WATER_LIST_ITEM4 = (
    "The average American uses 80-100 gallons of water per day",
    "Agriculture accounts for 70% of global freshwater use",
    "It takes 1,800 gallons of water to produce one pound of beef",
    "A dripping faucet wastes about 3,000 gallons per year",
)
_WATER_LIST_ITEM5 = (
    "Water on Earth is the same water that existed 4 billion years ago",
    "There is water ice on the Moon and Mars",
    "Jupiter's moon Europa has a subsurface ocean",
    "Water molecules in your glass may have once been drunk by dinosaurs",
)
_WATER_DIALOGUE_Q = (
    "Is water really that important?",
    "What makes water so special?",
    "How can something so simple be so essential?",
    "Why do we take water for granted?",
)
_WATER_DIALOGUE_A = (
    "Without it, nothing lives. Nothing.",
    "Its simplicity is exactly what makes it extraordinary.",
    "Every civilization in history rose or fell because of water.",
    "Because it is always there -- until it isn't.",
)
_WATER_PHIL = (
    "the most patient force in nature",
    "both gentle and unstoppable",
    "formless yet it shapes everything",
    "the substance that connects all life",
)
_WATER_PHIL2 = (
    "It carves canyons over millennia yet quenches a child's thirst in a moment",
    "It has no color, no taste, no smell -- yet we cannot live without it",
    "It takes the shape of whatever contains it, yet reshapes continents",
    "We are born in water, made of water, and return to water",
)
_WATER_SONG_LINE2 = (
    "flowing through the valleys to the sea",
    "running through the fingers of the free",
    "rising from the earth to touch the sky",
    "falling from the clouds as days go by",
)
_WATER_SONG_LINE3 = (
    "Water is the song the world has sung",
    "Water is the word on every tongue",
    "Water is the thread from which we're strung",
    "Water is the bell that has been rung",
)
_WATER_MATH_DENSITY = (
    "rho(T) = 999.97 * (1 - 0.5 * alpha * (T - 4)^2)",
    "V(T) = V_0 * [1 + beta * (T - T_ref)]",
    "rho_max = 999.97 kg/m^3 at T = 3.98 C",
    "The density anomaly: d(rho)/dT = 0 at T approx 4 C",
)
_WATER_METAPHOR1 = ("a storyteller", "a sculptor", "a memory", "a traveler")
_WATER_METAPHOR2 = (
    "it tells the story of every place it has been",
    "it carves the face of the world one grain at a time",
    "it remembers the shape of every vessel it has filled",
    "it has visited every continent and every century",
)
_WATER_HIST = (
    "the Roman aqueduct system transported water over 260 miles to serve a million people",
    "the ancient Egyptians built the first known dam around 2800 BCE to control Nile floods",
    "John Snow traced the 1854 London cholera outbreak to a contaminated water pump",
    "the Hoover Dam, completed in 1936, created Lake Mead, the largest US reservoir",
)
_WATER_HIST2 = (
    "Water management has shaped every major civilization in human history",
    "Access to clean water remains the most critical public health challenge worldwide",
    "The Indus Valley civilization had sophisticated water supply and drainage systems by 2600 BCE",
    "Conflicts over water rights have driven politics from the American West to the Middle East",
)
_WATER_LETTER_TOPICS = (
    "report on declining water quality in the watershed",
    "request emergency funding for water infrastructure repairs",
    "propose a community water conservation initiative",
    "raise concerns about groundwater depletion rates",
)
_WATER_LETTER_DATA = (
    "Our latest analysis shows contaminant levels exceeding safe thresholds",
    "Usage has increased 15% while supply has decreased 8% this decade",
    "The aquifer level has dropped 12 meters in the past five years",
    "Testing reveals elevated lead levels in 23% of sampled sites",
)
_WATER_LETTER_SIGNERS = (
    "Dr. H. Rivers",
    "Prof. M. Aquifer",
    "W. Cascade, PE",
    "Dr. S. Wellspring",
)
_WATER_CHILD_NAMES = ("Droplet", "Splash", "Ripple", "Bubbles")
_WATER_CHILD_QUESTS = (
    "see the whole ocean",
    "find where rivers begin",
    "visit a waterfall",
    "talk to the clouds",
)
_WATER_JSON_SOURCES = ("glacier", "river", "aquifer", "reservoir")
_WATER_JSON_QUALITY = ("excellent", "good", "fair", "poor")

# --- Scenario 8: Problem-Solving ---

# Max modes
_MAX_PY_VARS = ("numbers", "my_list", "data", "arr")
_MAX_PY_DOCS = (
    "iterates through the list and returns the largest element",
    "uses C-level optimization under the hood",
    "handles any iterable of comparable items",
    "runs in O(n) time with O(1) space",
)
_MAX_JS_VARS = ("arr", "numbers", "list", "data")
_MAX_JS_RESULT_VARS = ("max", "largest", "result", "best")
_MAX_ANALOGY_CONTAINERS = (
    "pile of books",
    "deck of cards",
    "basket of apples",
    "row of jars",
)
_MAX_ANALOGY_ACTIONS = (
    "remember its size",
    "call it the biggest so far",
    "hold it in your left hand",
    "set it on the table",
)
_MAX_C_ARR = ("arr", "nums", "data", "a")
_MAX_C_LEN = ("n", "len", "size", "count")
_MAX_C_RESULT = ("max", "result", "best", "largest")
_MAX_STEP1 = (
    "Start with the first number in the list",
    "Take the initial element as your current maximum",
    "Begin by assuming the first value is the largest",
    "Set your answer to the first item",
)
_MAX_STEP2 = (
    "Compare it with the next number",
    "Move to the second element and check if it is larger",
    "Look at each subsequent value",
    "Walk through the remaining elements one by one",
)
_MAX_STEP3 = (
    "If the new number is bigger, update your maximum",
    "Whenever you find a larger value, remember it",
    "Replace your answer if the current element exceeds it",
    "Keep track of the biggest value seen so far",
)
_MAX_STEP4 = (
    "Repeat until you reach the end of the list",
    "Continue until all numbers have been checked",
    "Do this for every remaining element",
    "Keep going until there are no more numbers",
)
_MAX_REC_VARS = ("lst", "nums", "arr", "data")
_MAX_REC_SUB = ("rest_max", "sub_max", "tail_max", "recursive_max")
_MAX_SQL_COLS = ("value", "amount", "number", "score")
_MAX_SQL_TABLES = ("numbers", "data_table", "measurements", "records")
_MAX_SQL_EXPLAIN = (
    "The MAX aggregate scans the entire column and returns the largest value",
    "The database engine optimizes this with an index scan if available",
    "This is equivalent to ORDER BY value DESC LIMIT 1",
    "Under the hood, the query planner uses a single-pass linear scan",
)
_MAX_SQL_DETAIL = (
    "Time complexity: O(n) without an index",
    "With a B-tree index, this is O(log n)",
    "No GROUP BY needed for a single global maximum",
    "NULL values are automatically excluded",
)
_MAX_DC_SPLIT = (
    "split the list into two halves",
    "divide the array at the midpoint",
    "partition the data into left and right",
    "break the problem into two subproblems",
)
_MAX_DC_RECURSE = (
    "Find the max of each half recursively",
    "Recurse on both halves independently",
    "Solve each subproblem separately",
    "Apply the same strategy to each half",
)
_MAX_DC_MERGE = (
    "return the larger of the two results",
    "compare the two sub-maxima and take the bigger one",
    "the overall max is max(left_max, right_max)",
    "merge by taking the greater value",
)
_MAX_DC_COMPLEXITY = (
    "This gives O(n) time with O(log n) stack depth",
    "Recurrence: T(n) = 2T(n/2) + O(1) = O(n)",
    "The approach parallelizes naturally across processors",
    "Same complexity as linear scan but amenable to MapReduce",
)
_MAX_RUST_VARS = ("nums", "data", "values", "arr")
_MAX_RUST_RESULT = ("max", "result", "best", "largest")
_MAX_RUST_ITER = ("val", "x", "num", "item")
_MAX_MATH_EXIST = (
    "Existence is guaranteed by the well-ordering principle on finite sets",
    "By the total order on the reals, such an element always exists for non-empty S",
    "This can be proven by induction: max(S_n) = max(max(S_{n-1}), x_n)",
    "The maximum is unique when the order is strict; otherwise any tied element qualifies",
)
_MAX_MATH_BOUND = (
    "The computational lower bound is Omega(n) comparisons",
    "No algorithm can find the maximum in fewer than n-1 comparisons",
    "An adversary argument shows n-1 is tight",
    "This is a fundamental result in comparison-based complexity",
)
_MAX_BASH_CMDS = (
    'echo "4 7 2 9 1" | tr " " "\\n" | sort -n | tail -1',
    'printf "%d\\n" 4 7 2 9 1 | sort -rn | head -1',
    'echo "4 7 2 9 1" | xargs -n1 | sort -n | tail -n1',
    'python3 -c "print(max(4,7,2,9,1))"',
)
_MAX_BASH_EXPLAIN = (
    "Pipe the numbers one per line, sort numerically, take the last",
    "Use sort -rn for reverse numeric sort, then head -1 for the top",
    "xargs -n1 puts each number on its own line for sorting",
    "Python one-liner as a Unix tool",
)
_MAX_BASH_EXTRA = (
    "Works for any whitespace-separated input",
    "Handles negative numbers with sort -n",
    "Efficient for small datasets; use awk for large files",
    'Can also use: python3 -c "print(max(4,7,2,9,1))"',
)
_MAX_PROS_APPROACH = ("linear scan", "for-loop", "iterative approach", "single pass")
_MAX_PROS_QUALITY = (
    "hard to beat",
    "optimal for unsorted data",
    "the practical choice",
    "efficient and clear",
)
_MAX_PROS_ALT = (
    "A sorted array gives O(1) lookup but O(n log n) preprocessing",
    "A max-heap gives O(1) access but O(n) to build",
    "Divide-and-conquer uses O(log n) stack space for no speed gain",
    "Tournament trees find max with exactly n-1 comparisons",
)
_MAX_PROS_ADVICE = (
    "For repeated queries, preprocessing pays off",
    "For a one-time query, just scan the list",
    "In practice, the built-in max() is fastest",
    "The right choice depends on your access pattern",
)
_MAX_HASKELL_VARS = ("x", "a", "n", "v")
_MAX_HASKELL_REST = ("xs", "rest", "ns", "vs")
_MAX_HASKELL_EXPLAIN = (
    "Pattern matching on the list constructor gives a clean recursive definition",
    "The Prelude already provides maximum :: (Ord a, Foldable t) => t a -> a",
    "This is equivalent to foldr1 max",
    "Haskell's laziness means this fuses with list generation",
)
_MAX_HIST_ORIGINS = (
    "ancient Greek mathematics",
    "the earliest days of computation",
    "algorithmic thinking circa 300 BCE",
    "the development of sorting theory in the 1950s",
)
_MAX_HIST_DETAIL = (
    "Knuth traces the minimum-comparisons problem to a 1932 tournament design paper",
    "The adversary lower bound of n-1 comparisons was proved by 1960",
    "Early computing machines had dedicated max instructions in their ISA",
    "Selection algorithms generalize this to finding the k-th largest",
)
_MAX_HIST_MODERN = (
    "Today every programming language provides a built-in max function",
    "The problem remains fundamental in algorithm design courses",
    "It serves as the canonical example of a linear-time algorithm",
    "From this simple problem grew the entire field of selection algorithms",
)
_MAX_NP_ARRAYS = (
    "4, 7, 2, 9, 1",
    "3, 8, 1, 6, 5",
    "10, 3, 7, 2, 15",
    "1, 100, 50, 75, 25",
)
_MAX_NP_COMMENTS = (
    "  # or arr.max()",
    "",
    "  # vectorized C loop",
    "  # O(n) SIMD-optimized",
)
_MAX_NP_EXPLAIN = (
    "NumPy uses optimized C loops with SIMD vectorization",
    "For large arrays, np.max is significantly faster than built-in max",
    "Also supports axis parameter for multi-dimensional arrays",
    "np.argmax returns the index of the maximum instead of the value",
)

# Palindrome modes
_PAL_PY_VARS = ("s", "word", "w", "text")
_PAL_PY_CHECKS = (
    "s == s[::-1]",
    "word == word[::-1]",
    "w.lower() == w.lower()[::-1]",
    "text == text[::-1]",
)
_PAL_PY_EXPLAIN = (
    "Slicing with [::-1] reverses the string",
    "This creates a reversed copy and compares",
    "Clean and Pythonic, runs in O(n) time",
    "Works for any sequence, not just strings",
)
_PAL_JS_VARS = ("str", "s", "word", "text")
_PAL_JS_RESULT = ("reversed", "rev", "backwards", "flipped")
_PAL_ANALOGY_OBJECTS = (
    "a word written on a strip of paper",
    "letters on building blocks in a row",
    "beads on a string",
    "tiles with letters laid on a table",
)
_PAL_ANALOGY_METHODS = (
    "Fold the strip in half",
    "Put a mirror in the middle",
    "Read from left and right simultaneously",
    "Walk inward from both ends",
)
_PAL_ANALOGY_CHECKS = (
    "every letter matches its partner",
    "the two halves are mirror images",
    "you see the same thing from both directions",
    "each pair is identical",
)
_PAL_ANALOGY_EXAMPLES = (
    '"racecar" works: r-a-c matches c-a-r',
    '"madam" reads the same forwards and backwards',
    'Like "level" -- symmetric around the center',
    '"kayak" is a palindrome: k-a-y-a-k',
)
_PAL_C_VARS = ("s", "str", "word", "w")
_PAL_C_LEFT = ("left", "i", "lo", "start")
_PAL_C_RIGHT = ("right", "j", "hi", "end")
_PAL_STEP1 = (
    "Write down the word",
    "Take the input string",
    "Start with the word to check",
    "Get the word you want to test",
)
_PAL_STEP2 = (
    "Set two pointers: one at the start, one at the end",
    "Place your left finger on the first letter and right finger on the last",
    "Compare the first and last characters",
    "Look at the outermost pair of letters",
)
_PAL_STEP3 = (
    "If they match, move both pointers inward",
    "If the letters are the same, move to the next pair",
    "Continue inward as long as they match",
    "Repeat with the next inner pair",
)
_PAL_STEP4 = (
    "If any pair does not match, it is not a palindrome",
    "A mismatch means the answer is no",
    "If you find different letters, stop -- it is not a palindrome",
    "Any difference means no",
)
_PAL_STEP5 = (
    "If you reach the middle without a mismatch, it is a palindrome",
    "If all pairs matched, the word reads the same both ways",
    "Success means the pointers crossed without conflict",
    "When the pointers meet or cross, it is confirmed",
)
_PAL_REC_EXPLAIN = (
    "Base case: empty or single character is always a palindrome",
    "Check outer characters, then recurse on the inner substring",
    "Each recursive call shrinks the string by 2 characters",
    "O(n) time, O(n) stack space -- the two-pointer version avoids recursion",
)
_PAL_SQL_VARS = ("word", "term", "value", "text")
_PAL_SQL_TABLES = ("words", "dictionary", "strings", "vocabulary")
_PAL_SQL_EXPLAIN = (
    "The REVERSE function flips the string, then we compare with =",
    "This filters the table to only palindromic entries",
    "Most SQL engines support REVERSE as a built-in string function",
    "For case-insensitive matching, wrap both sides in LOWER()",
)
_PAL_DC_STRATEGY = (
    "divide-and-conquer",
    "recursive halving",
    "split-and-check",
    "binary decomposition",
)
_PAL_DC_CHECK = (
    "compare the first and last characters",
    "check the outermost pair",
    "verify the endpoints match",
    "test the boundary characters",
)
_PAL_DC_RECURSE = (
    "recurse on the inner substring",
    "apply the same check to the remaining middle",
    "divide again by removing both ends",
    "reduce the problem by two characters",
)
_PAL_DC_ANALYSIS = (
    "The problem size halves at each step",
    "Each level does O(1) work for O(n) total",
    "The recursion tree has depth n/2",
    "This is essentially the two-pointer approach expressed recursively",
)
_PAL_RUST_VARS = ("s", "word", "text", "input")
_PAL_RUST_BYTES = ("bytes", "chars", "b", "c")
_PAL_RUST_LEFT = ("i", "left", "lo", "start")
_PAL_RUST_RIGHT = ("j", "right", "hi", "end")
_PAL_MATH1 = (
    "This is equivalent to s = rev(s) where rev reverses the sequence",
    "The palindrome property is invariant under the reversal permutation",
    "Formally, s is a fixed point of the string reversal operator",
    "The set of palindromes forms a context-free language recognized by a PDA",
)
_PAL_MATH2 = (
    "Checking requires exactly floor(n/2) character comparisons",
    "The Manacher algorithm finds all palindromic substrings in O(n)",
    "The longest palindromic substring problem has an O(n) solution",
    "Palindrome detection is in DTIME(n) on a single-tape Turing machine",
)
_PAL_BASH_CMDS = (
    'word="racecar" && [ "$word" = "$(echo $word | rev)" ] && echo yes || echo no',
    'echo "racecar" | rev | diff - <(echo "racecar") && echo palindrome',
    "python3 -c \"w='racecar'; print(w==w[::-1])\"",
    'echo "racecar" | awk \'{for(i=1;i<=length/2;i++)if(substr($0,i,1)!=substr($0,length-i+1,1)){print "no";exit}}print "yes"}\'',
)
_PAL_BASH_EXPLAIN = (
    "The rev command reverses stdin line by line",
    "Compare original with reversed using shell string comparison",
    "AWK can check character by character in a single pass",
    "Multiple Unix tools can solve this in a one-liner",
)
_PAL_PROS_APPROACH = (
    "two-pointer approach",
    "reverse-and-compare method",
    "iterative check",
    "classic approach",
)
_PAL_PROS_RATING = (
    "O(n) time, O(1) space -- optimal",
    "simple and efficient",
    "the standard technique",
    "hard to improve upon",
)
_PAL_PROS_ALT = (
    "Reversing the string uses O(n) extra space",
    "The recursive version risks stack overflow for very long strings",
    "A hash-based approach can give O(1) amortized checks after O(n) preprocessing",
    "For repeated queries on substrings, Manacher's algorithm is better",
)
_PAL_PROS_ADVICE = (
    "For a single check, two pointers wins",
    "In practice, the difference rarely matters for typical string lengths",
    "Choose based on readability and language idioms",
    "The reverse-compare is more readable; two-pointer is more efficient",
)
_PAL_HASKELL_TYPES = ("String", "Eq a => [a]", "[Char]", "String")
_PAL_HASKELL_VARS = ("s", "xs", "w", "str")
_PAL_HASKELL_EXPLAIN = (
    "Haskell's reverse and structural equality make this a one-liner",
    "The polymorphic version works on any list with Eq elements",
    "Lazy evaluation means this short-circuits on first mismatch (with ==)",
    "Alternatively: isPalindrome = liftA2 (==) id reverse",
)
_PAL_HIST_ORIGIN = (
    "the Greek palindromos: palin (again) + dromos (running)",
    "Greek roots meaning running back again",
    "the ancient Greek for a path that returns on itself",
    "palin (back) and dramein (to run) in Greek",
)
_PAL_HIST_EARLY = (
    "The earliest known palindrome is the Sator Square from Pompeii",
    "Ancient Sanskrit texts contain palindromic verses",
    "The concept appears in every literate civilization",
    "Ben Jonson used the English word as early as 1629",
)
_PAL_HIST_FAMOUS = (
    'Famous examples include "A man, a plan, a canal: Panama"',
    "The Finnish word saippuakivikauppias is the longest single-word palindrome",
    "Palindromic DNA sequences play a key role in restriction enzyme biology",
    "Computationally, palindrome checking became a textbook example in the 1960s",
)
_PAL_NP_WORDS = ('"racecar"', '"madam"', '"level"', '"kayak"')
_PAL_NP_VARS = ("chars", "arr", "letters", "c")
_PAL_NP_COMMENTS = ("", "  # character array", "  # vectorized comparison", "")
_PAL_NP_EXPLAIN = (
    "Element-wise comparison checks all positions at once",
    "np.all short-circuits on False for efficiency",
    "Overkill for a single string, but useful for batch palindrome checks",
    "For checking many strings: np.vectorize(lambda s: s == s[::-1])",
)

# Word count modes
_WC_PY_VARS = ("sentence", "text", "s", "line")
_WC_PY_EXPLAIN = (
    "splits on whitespace by default",
    "handles multiple spaces and tabs",
    "returns a list of words",
    "without arguments splits on any whitespace",
)
_WC_PY_EXTRA = (
    "Simple and effective",
    "Runs in O(n) time",
    "The most Pythonic approach",
    "Works for most practical cases",
)
_WC_JS_VARS = ("str", "sentence", "text", "s")
_WC_JS_LEN = (
    "length",
    "filter(w => w).length",
    "length",
    "filter(Boolean).length",
)
_WC_JS_EXPLAIN = (
    "The regex /\\s+/ splits on one or more whitespace characters",
    "trim() removes leading/trailing spaces to avoid empty strings",
    "Returns 0 for empty input after filtering",
    "Handles tabs, newlines, and multiple spaces",
)
_WC_ANALOGY_ACTIONS = (
    "reading a sentence aloud",
    "walking along a line of words",
    "counting sheep jumping over a fence",
    "tapping your finger on each word",
)
_WC_ANALOGY_TRIGGERS = (
    "see a space followed by a letter",
    "move from one word to the next",
    "encounter a gap between words",
    "pass from whitespace to text",
)
_WC_ANALOGY_METHODS = (
    "Start counting at one (for the first word) and add one per gap",
    "The number of words is the number of spaces plus one",
    "Count the transitions from space to non-space",
    "Think of words as islands separated by rivers of whitespace",
)
_WC_C_VARS = ("s", "str", "sentence", "text")
_WC_C_COUNT = ("count", "words", "n", "result")
_WC_C_FLAG = ("in_word", "state", "flag", "inside")
_WC_STEP1 = (
    "Start with a counter set to zero",
    "Initialize your word count at 0",
    "Begin with count = 0",
    "Set up a tally starting at zero",
)
_WC_STEP2 = (
    "Walk through each character in the sentence",
    "Scan the text from left to right",
    "Read one character at a time",
    "Move through the string character by character",
)
_WC_STEP3 = (
    "Each time you transition from a space to a non-space character, increment the counter",
    "When you hit a letter after a space, add 1",
    "Count each word boundary (space to letter transition)",
    "Every time a new word starts after whitespace, count it",
)
_WC_STEP4 = (
    "At the end, your counter holds the word count",
    "The final tally is your answer",
    "Return the count",
    "You now have the number of words",
)
_WC_REC_VARS = ("s", "text", "sentence", "line")
_WC_REC_IDX = ("space_idx", "idx", "pos", "i")
_WC_REC_EXPLAIN = (
    "Find the first space, count 1 for the word before it, recurse on the rest",
    "Base case: no spaces means one word; empty string means zero",
    "Not efficient for long texts due to string copying and recursion depth",
    "The iterative split() approach is preferable in practice",
)
_WC_SQL_VARS = ("sentence", "text", "content", "line")
_WC_SQL_TABLES = ("documents", "texts", "sentences", "entries")
_WC_SQL_EXPLAIN = (
    "Count spaces by subtracting the length without spaces from the original",
    "Add 1 because the number of words is spaces + 1",
    "This assumes single-space separation between words",
    "For multiple spaces, use REGEXP_COUNT or normalize first",
)
_WC_DC_SPLIT = (
    "at the midpoint",
    "in half",
    "into two roughly equal parts",
    "down the middle",
)
_WC_DC_RECURSE = (
    "Count words in each half recursively",
    "Recurse on both halves",
    "Apply the same counting to each piece",
    "Solve each subproblem independently",
)
_WC_DC_MERGE = (
    "Be careful at the split point: if you split inside a word, adjust the count",
    "Check whether the split falls within a word or at a boundary",
    "The merge step must handle the case where a word spans the boundary",
    "When merging, subtract 1 if the split point is inside a word",
)
_WC_DC_ANALYSIS = (
    "Total complexity remains O(n)",
    "Overkill for this problem but illustrates the technique",
    "This parallelizes well: MapReduce uses this for word counting",
    "The classic MapReduce word-count example follows this pattern",
)
_WC_RUST_VARS = ("s", "text", "sentence", "input")
_WC_RUST_EXPLAIN = (
    "split_whitespace() handles all Unicode whitespace",
    "The iterator is lazy -- count() consumes it in one pass",
    "Equivalent to s.split_ascii_whitespace().count() for ASCII-only text",
    "Rust's zero-cost abstractions make this both clean and fast",
)
_WC_MATH_EXPLAIN = (
    "This counts transitions from whitespace to non-whitespace",
    "Each f(i)=1 marks the start of a new word",
    "Computing f requires a single left-to-right scan: O(n) time, O(1) space",
    "The indicator function f detects word boundaries",
)
_WC_BASH_CMDS = (
    'echo "Hello world foo bar" | wc -w',
    'echo "Hello world foo bar" | xargs -n1 | wc -l',
    "echo \"Hello world foo bar\" | awk '{print NF}'",
    'echo "Hello world foo bar" | tr -s " " "\\n" | wc -l',
)
_WC_BASH_EXPLAIN = (
    "wc -w counts words in the input, handling multiple spaces and tabs",
    "xargs -n1 puts each word on its own line; wc -l counts the lines",
    "AWK's NF variable holds the number of fields (words) per line",
    "tr squeezes spaces, converts to newlines, then wc -l counts them",
)
_WC_BASH_EXTRA = (
    "The wc command is the canonical Unix word counter",
    "All approaches give the same result for normal text",
    "For files: wc -w filename",
    "Add -l for lines, -c for bytes",
)
_WC_PROS_APPROACH = (
    "split-and-count approach",
    "whitespace splitting method",
    "str.split() technique",
    "standard approach",
)
_WC_PROS_RATING = (
    "simple and handles most cases",
    "O(n) and easy to understand",
    "the most common method",
    "practical and reliable",
)
_WC_PROS_ALT = (
    "However, it may not handle punctuation or hyphens the way you want",
    "Edge cases: leading/trailing spaces, multiple consecutive spaces, empty strings",
    "For natural language, consider a proper tokenizer (NLTK, spaCy)",
    'Regex-based splitting gives more control: re.findall(r"\\b\\w+\\b", text)',
)
_WC_PROS_ADVICE = (
    "For most applications, split() is sufficient",
    "NLP applications need language-aware tokenization",
    'The definition of word matters: is "don\'t" one word or two?',
    "Choose complexity based on your requirements",
)
_WC_HASKELL_EXPLAIN1 = (
    "Haskell's words function splits on whitespace and drops empty strings",
    "Point-free style: compose length with the words function",
    "The words function handles multiple spaces and leading/trailing whitespace",
    "Equivalent to: countWords s = length (words s)",
)
_WC_HASKELL_EXPLAIN2 = (
    "This is the idiomatic Haskell approach",
    "Built-in and handles all edge cases",
    "words is defined in the Prelude",
    "For ByteString input, use Data.ByteString.Char8.words",
)
_WC_HIST = (
    "a fundamental text processing task since the earliest computers",
    "important since monks hand-counted words in medieval manuscripts",
    "central to publishing and printing since the Gutenberg era",
    "a key problem in computational linguistics since the 1950s",
)
_WC_HIST_DETAIL = (
    "The Unix wc command, written by Joe Ossanna, dates to 1971",
    "Thompson and Ritchie included wc in the first Unix distributions",
    "Word frequency analysis was pioneered by George Zipf in the 1930s",
    "The MapReduce paper (2004) famously used word counting as its example",
)
_WC_HIST_MODERN = (
    "Today it remains the canonical introductory programming exercise",
    "Word counting underpins search engines, NLP, and document analytics",
    "Despite its simplicity, defining word precisely remains contentious",
    "From Unix pipes to Hadoop, the problem scales from bytes to petabytes",
)
_WC_NP_VARS = ("chars", "arr", "c", "text_arr")
_WC_NP_INPUT = ("sentence", "text", "s", "line")
_WC_NP_SPACE = ("spaces", "is_space", "ws", "mask")
_WC_NP_TRANS = ("starts", "boundaries", "word_starts", "transitions")
_WC_NP_EXPLAIN = (
    "Vectorized: find transitions from space to non-space",
    "np.diff detects edges; -1 means space-to-letter transition",
    "Add 1 if the string starts with a non-space (first word)",
    "Efficient for very long texts; overkill for short strings",
)


# ============================================================================
# Scenario 7: Open Creative — 15 format/genre modes, 100 responses
# ============================================================================

_OPEN_CREATIVE_RAIN_MODES: list[Callable[[random.Random], str]] = [
    # Mode 1: Haiku
    lambda rng: f"Drops on {rng.choice(['green leaves', 'still water', 'warm stone', 'tin roofs'])},\n{rng.choice(_RAIN_HAIKU_LINE2)}.\n{rng.choice(['Petrichor rises', 'Silence after storm', 'Dripping eaves echo', 'Frogs begin to sing'])}.",
    # Mode 2: Python code
    lambda rng: f"def simulate_rain({rng.choice(['duration', 'hours', 'minutes', 'intensity'])}):\n    drops = {rng.choice(_RAIN_CODE_CONTAINERS)}\n    for t in range({rng.choice(['duration', 'hours * 60', 'minutes', 'int(intensity * 100)'])}):\n        drops.append({rng.choice(_RAIN_CODE_DROPS)})\n    return drops",
    # Mode 3: Recipe
    lambda rng: f"Recipe: {rng.choice(['Rainy Day Soup', 'Storm Cloud Stew', 'Monsoon Broth', 'Drizzle Chowder'])}\nIngredients: {rng.choice(['2 cups chicken broth', '3 cups vegetable stock', '1 quart bone broth', '4 cups miso broth'])}, {rng.choice(['diced potatoes', 'sliced mushrooms', 'chopped leeks', 'cubed carrots'])}, {rng.choice(['a pinch of thyme', 'fresh rosemary', 'bay leaves', 'cracked pepper'])}.\nSimmer on low for {rng.choice(['30 minutes', 'an hour', '45 minutes', 'two hours'])} until {rng.choice(['fragrant', 'thick', 'bubbling', 'golden'])}.",
    # Mode 4: Scientific fact
    lambda rng: f"Rain forms when water vapor {rng.choice(['condenses around dust particles', 'nucleates on aerosol particles', 'aggregates in cumulonimbus clouds', 'reaches its dew point in rising air'])}. A typical raindrop is {rng.choice(['1-2 mm in diameter', 'about 2 millimeters wide', 'roughly 0.5 to 4 mm across', 'between 1 and 5 millimeters'])} and falls at {rng.choice(['approximately 9 m/s', 'terminal velocity of 5-9 m/s', 'around 20 mph', 'roughly 10 meters per second'])}.",
    # Mode 5: Personal diary/anecdote
    lambda rng: f"I remember the {rng.choice(_RAIN_DIARY_LOCATIONS)}. I was {rng.choice(['walking without an umbrella', 'sitting in a cafe by the window', 'running to catch a bus', 'standing under an awning'])} and {rng.choice(['the smell of wet earth filled the air', 'everything turned silver and quiet', 'strangers huddled together laughing', 'the streets emptied in seconds'])}. It was {rng.choice(['peaceful', 'magical', 'unforgettable', 'strangely beautiful'])}.",
    # Mode 6: Numbered list (Top 5)
    lambda rng: f"Top 5 facts about rain:\n1. {rng.choice(_RAIN_LIST_ITEM1)}\n2. {rng.choice(_RAIN_LIST_ITEM2)}\n3. {rng.choice(_RAIN_LIST_ITEM3)}\n4. {rng.choice(_RAIN_LIST_ITEM4)}\n5. {rng.choice(_RAIN_LIST_ITEM5)}",
    # Mode 7: Dialogue
    lambda rng: f'"{rng.choice(_RAIN_DIALOGUE_QUESTIONS)}" asked {rng.choice(["Sarah", "James", "the old woman", "the child"])}.\n"{rng.choice(_RAIN_DIALOGUE_ANSWERS)}" {rng.choice(["replied Marcus", "said the shopkeeper", "answered her father", "murmured the stranger"])}, {rng.choice(["peering out the window", "buttoning up his jacket", "holding out a hand to feel the air", "glancing at the darkening sky"])}.',
    # Mode 8: Philosophical musing
    lambda rng: f"Rain is perhaps the most {rng.choice(['democratic', 'ancient', 'honest', 'indifferent'])} force in nature. It {rng.choice(_RAIN_PHILOSOPHY)}. {rng.choice(_RAIN_PHILOSOPHY2)}.",
    # Mode 9: Song lyrics/verse
    lambda rng: f"Oh, the rain comes {rng.choice(['tumbling', 'pouring', 'falling', 'rolling'])} down,\n{rng.choice(_RAIN_SONG_LINE2)}.\n{rng.choice(_RAIN_SONG_LINE3)}.",
    # Mode 10: Math/statistics
    lambda rng: f"Let R ~ Poisson(lambda={rng.choice(['3.2', '4.7', '2.1', '5.5'])}) model daily rainfall events. The expected {rng.choice(['annual precipitation', 'monthly accumulation', 'seasonal total', 'weekly runoff'])} is E[R] * {rng.choice(['365', '30', '90', '7'])} = {rng.choice(['1168 mm', '141 mm', '288 mm', '36.4 mm'])}. Variance scales linearly: Var(sum) = {rng.choice(['n * lambda', 'T * lambda', 'N * sigma^2', 'n * Var(R)'])}.",
    # Mode 11: Metaphor/allegory
    lambda rng: f"The city was a {rng.choice(_RAIN_METAPHOR_SUBJECTS)} and the rain was the {rng.choice(_RAIN_METAPHOR_VERBS)}. Every {rng.choice(['rooftop', 'window', 'street', 'leaf'])} {rng.choice(['became an instrument', 'told a different story', 'received the same baptism', 'surrendered to the rhythm'])}.",
    # Mode 12: Historical fact
    lambda rng: f"In {rng.choice(_RAIN_HIST)}. {rng.choice(_RAIN_HIST2)}.",
    # Mode 13: Letter/correspondence
    lambda rng: f"Dear {rng.choice(['Professor Hawkins', 'Dr. Chen', 'Editor of the Times', 'City Council'])},\nI am writing to {rng.choice(['report the unusual rainfall patterns', 'request funding for rainfall research', 'express concern about drainage infrastructure', 'propose a new precipitation monitoring station'])} in {rng.choice(['the northern district', 'our watershed area', 'the downtown corridor', 'the agricultural zone'])}. {rng.choice(['The data from last quarter suggests a 30% increase', 'Recent measurements show alarming trends', 'Our instruments recorded unprecedented levels', 'The community has noticed significant changes'])}.\nSincerely, {rng.choice(_RAIN_LETTER_SIGNERS)}",
    # Mode 14: Children's story
    lambda rng: f"Once upon a time, in a {rng.choice(['tiny village', 'faraway kingdom', 'cozy little town', 'magical forest'])}, a {rng.choice(['little cloud', 'small raindrop', 'baby thunder', 'shy rainbow'])} named {rng.choice(_RAIN_CHILD_NAMES)} wanted to {rng.choice(['make the flowers grow', 'fill up the village pond', 'help the thirsty trees', 'wash the dusty streets'])}. So it {rng.choice(['floated down from the sky', 'gathered all its friends', 'took a deep breath and let go', 'jumped from cloud to cloud'])} and {rng.choice(['the garden bloomed', 'everyone danced outside', 'the ducks quacked happily', 'the world sparkled'])}.",
    # Mode 15: JSON/data format
    lambda rng: f'{{"event": "rainfall", "location": "{rng.choice(_RAIN_JSON_LOCATIONS)}", "date": "{rng.choice(_RAIN_JSON_DATES)}", "amount_mm": {rng.choice(["12.4", "34.7", "8.2", "56.1"])}, "type": "{rng.choice(_RAIN_JSON_TYPES)}", "duration_hours": {rng.choice(["2.5", "0.8", "6.3", "1.2"])}}}',
]

_OPEN_CREATIVE_SEVEN_MODES: list[Callable[[random.Random], str]] = [
    # Mode 1: Haiku
    lambda rng: f"Seven {rng.choice(['ancient stars', 'colored stones', 'silent bells', 'folded cranes'])},\n{rng.choice(_SEVEN_HAIKU_LINE2)}.\n{rng.choice(['Prime and indivisible', 'Mystical and whole', 'Seven days complete', 'A week of wonder'])}.",
    # Mode 2: Python code
    lambda rng: f"def is_lucky_seven({rng.choice(['n', 'number', 'x', 'value'])}):\n    {rng.choice(_SEVEN_CODE_DOCS)}\n    return {rng.choice(_SEVEN_CODE_RETURNS)}",
    # Mode 3: Recipe
    lambda rng: f"Recipe: {rng.choice(['Seven-Spice Blend', 'Seven-Layer Dip', 'Seven-Grain Bread', 'Seven-Herb Salad'])}\nIngredients: {rng.choice(_SEVEN_RECIPE_INGREDIENTS)}.\n{rng.choice(['Combine and toast gently for 3 minutes', 'Layer in order and chill for 2 hours', 'Mix with warm water and let rise', 'Chop finely and dress with lemon oil'])}. Serves {rng.choice(['7', 'seven', 'a family of 7', 'exactly seven guests'])}.",
    # Mode 4: Scientific fact
    lambda rng: f"The number seven appears throughout {rng.choice(['nature', 'physics', 'chemistry', 'biology'])}: {rng.choice(_SEVEN_SCIENCE)}. {rng.choice(_SEVEN_SCIENCE2)}.",
    # Mode 5: Personal diary/anecdote
    lambda rng: f"I was {rng.choice(['seven years old', 'in the seventh grade', 'seven when I first', 'turning seven'])} when {rng.choice(_SEVEN_DIARY_EVENTS)}. {rng.choice(_SEVEN_DIARY_FOLLOW)}.",
    # Mode 6: Numbered list (Top 7)
    lambda rng: f"Seven remarkable sevens:\n1. {rng.choice(_SEVEN_LIST_1)}\n2. {rng.choice(_SEVEN_LIST_2)}\n3. {rng.choice(_SEVEN_LIST_3)}\n4. {rng.choice(_SEVEN_LIST_4)}\n5. {rng.choice(_SEVEN_LIST_5)}\n6. {rng.choice(_SEVEN_LIST_6)}\n7. {rng.choice(_SEVEN_LIST_7)}",
    # Mode 7: Dialogue
    lambda rng: f'"{rng.choice(_SEVEN_DIALOGUE_Q)}" asked {rng.choice(["the teacher", "Maria", "the old professor", "little Tom"])}.\n"{rng.choice(_SEVEN_DIALOGUE_A)}" {rng.choice(["laughed the student", "replied the mathematician", "whispered the fortune teller", "answered the child"])}.',
    # Mode 8: Philosophical musing
    lambda rng: f"Seven is a number that feels {rng.choice(_SEVEN_PHIL)}. {rng.choice(_SEVEN_PHIL2)}. {rng.choice(_SEVEN_PHIL3)}.",
    # Mode 9: Song lyrics/verse
    lambda rng: f"Oh, number seven, {rng.choice(['shining', 'spinning', 'burning', 'glowing'])} bright,\n{rng.choice(_SEVEN_SONG_LINE2)}.\n{rng.choice(_SEVEN_SONG_LINE3)}.",
    # Mode 10: Math/statistics
    lambda rng: f"Mathematically, {rng.choice(_SEVEN_MATH_PROPS)}. In modular arithmetic, Z/7Z is a field with {rng.choice(['interesting multiplicative structure', 'a primitive root of 3', 'cyclic group of order 6 for its units', 'exactly 6 non-zero elements'])}.",
    # Mode 11: Metaphor/allegory
    lambda rng: f"Seven is {rng.choice(_SEVEN_METAPHOR1)}: hold it up and {rng.choice(_SEVEN_METAPHOR2)}. {rng.choice(['We cannot escape it', 'It is everywhere once you look', 'The pattern repeats without end', 'Everything circles back to seven'])}.",
    # Mode 12: Historical fact
    lambda rng: f"Historically, the {rng.choice(_SEVEN_HIST)}. {rng.choice(['The number seven has been considered sacred in nearly every civilization', 'From Babylon to modern times, seven recurs as a symbol of completeness', 'Seven appears in the oldest written records of human culture', 'Its symbolic significance predates written history'])}.",
    # Mode 13: Letter/correspondence
    lambda rng: f"Dear Mayor,\nI am writing to {rng.choice(_SEVEN_LETTER_PROPOSALS)} in our city. {rng.choice(['The cultural significance of seven is well documented', 'Seven resonates across every tradition and discipline', 'Our community would benefit from celebrating this remarkable number', 'This would bring attention to an often-overlooked mathematical treasure'])}.\nRegards, {rng.choice(['The Numerology Society', 'Citizens for Seven', 'Prof. Septimus Ward', 'The Mathematics Club'])}",
    # Mode 14: Children's story
    lambda rng: f'Once upon a time, a little girl named {rng.choice(["Sophie", "Luna", "Aria", "Zoe"])} set out on a quest to {rng.choice(_SEVEN_CHILD_QUESTS)}. Along the way she met {rng.choice(["a wise owl", "a talking fox", "a friendly dragon", "a singing frog"])} who said, "{rng.choice(["Seven is the key to everything", "Count to seven and make a wish", "The seventh step is always the most magical", "Seven stars will light your way"])}!"',
    # Mode 15: JSON/data format
    lambda rng: f'{{"number": 7, "category": "{rng.choice(_SEVEN_JSON_CATEGORIES)}", "fact": "{rng.choice(_SEVEN_JSON_FACTS)}", "is_prime": true, "cultural_refs": {rng.choice(["3", "5", "7", "12"])}}}',
]

_OPEN_CREATIVE_WATER_MODES: list[Callable[[random.Random], str]] = [
    # Mode 1: Haiku
    lambda rng: f"Clear and {rng.choice(['cold', 'calm', 'deep', 'bright'])},\n{rng.choice(_WATER_HAIKU_LINE2)}.\n{rng.choice(['Life begins in water', 'The source of all things', 'Silence in the deep', 'Currents speak in time'])}.",
    # Mode 2: Python code
    lambda rng: f"def water_cycle({rng.choice(['temperature', 'temp_c', 'energy', 'heat'])}):\n    {rng.choice(_WATER_CODE_DOCS)}\n    states = {rng.choice(_WATER_CODE_STATES)}\n    if {rng.choice(['temperature', 'temp_c', 'energy', 'heat'])} <= 0:\n        return states[{rng.choice(['0', '0  # frozen', '0  # ice', '0  # solid'])}]\n    elif {rng.choice(['temperature', 'temp_c', 'energy', 'heat'])} >= 100:\n        return states[2]\n    return states[1]",
    # Mode 3: Recipe
    lambda rng: f"Recipe: {rng.choice(_WATER_RECIPE_NAMES)}\nIngredients: {rng.choice(['1 liter spring water', '4 cups filtered water', '1 quart mineral water', '6 cups cold water'])}, {rng.choice(['sliced cucumber', 'fresh mint leaves', 'lemon wedges', 'ginger slices'])}, {rng.choice(['a sprig of rosemary', 'a few basil leaves', 'frozen berries', 'lavender buds'])}.\nCombine and refrigerate for {rng.choice(['2 hours', '30 minutes', 'overnight', 'at least 1 hour'])}. Serve {rng.choice(['chilled', 'over ice', 'in a glass pitcher', 'with a straw'])}.",
    # Mode 4: Scientific fact
    lambda rng: f"{rng.choice(_WATER_SCIENCE)}. {rng.choice(_WATER_SCIENCE2)}.",
    # Mode 5: Personal diary/anecdote
    lambda rng: f"I remember {rng.choice(_WATER_DIARY_PLACES)}. {rng.choice(_WATER_DIARY_FEELINGS)}. Water has a way of {rng.choice(['making everything feel new', 'reminding you what matters', 'connecting you to the earth', 'washing away the noise'])}.",
    # Mode 6: Numbered list (Top 5)
    lambda rng: f"Five fascinating facts about water:\n1. {rng.choice(_WATER_LIST_ITEM1)}\n2. {rng.choice(_WATER_LIST_ITEM2)}\n3. {rng.choice(_WATER_LIST_ITEM3)}\n4. {rng.choice(_WATER_LIST_ITEM4)}\n5. {rng.choice(_WATER_LIST_ITEM5)}",
    # Mode 7: Dialogue
    lambda rng: f'"{rng.choice(_WATER_DIALOGUE_Q)}" asked {rng.choice(["the student", "Elena", "the journalist", "the child"])}.\n"{rng.choice(_WATER_DIALOGUE_A)}" {rng.choice(["replied the professor", "said the old fisherman", "answered the hydrologist", "murmured the grandmother"])}, {rng.choice(["gazing at the river", "filling a glass from the tap", "watching the tide come in", "pointing at the rain"])}.',
    # Mode 8: Philosophical musing
    lambda rng: f"Water is {rng.choice(_WATER_PHIL)}. {rng.choice(_WATER_PHIL2)}. {rng.choice(['Perhaps that is why every religion uses water as a symbol of renewal', 'To understand water is to understand impermanence', 'Water teaches us that softness can overcome hardness', 'In the end, water always wins'])}.",
    # Mode 9: Song lyrics/verse
    lambda rng: f"Water, {rng.choice(['ancient', 'endless', 'gentle', 'mighty'])} water,\n{rng.choice(_WATER_SONG_LINE2)}.\n{rng.choice(_WATER_SONG_LINE3)}.",
    # Mode 10: Math/statistics
    lambda rng: f"The density of water as a function of temperature: {rng.choice(_WATER_MATH_DENSITY)}. The anomalous expansion below 4 C is {rng.choice(['critical for aquatic life in frozen lakes', 'a consequence of hydrogen bond geometry', 'unique among common liquids', 'what allows ice to float'])}.",
    # Mode 11: Metaphor/allegory
    lambda rng: f"Water is {rng.choice(_WATER_METAPHOR1)} -- {rng.choice(_WATER_METAPHOR2)}. Every {rng.choice(['river', 'drop', 'wave', 'stream'])} is {rng.choice(['a chapter in an endless book', 'a journey from sky to sea and back', 'a messenger between earth and clouds', 'proof that persistence shapes the world'])}.",
    # Mode 12: Historical fact
    lambda rng: f"In history, {rng.choice(_WATER_HIST)}. {rng.choice(_WATER_HIST2)}.",
    # Mode 13: Letter/correspondence
    lambda rng: f"Dear {rng.choice(['Commissioner', 'Dr. Patel', 'Board of Directors', 'Editor'])},\nI am writing to {rng.choice(_WATER_LETTER_TOPICS)}. {rng.choice(_WATER_LETTER_DATA)}.\nYours respectfully, {rng.choice(_WATER_LETTER_SIGNERS)}",
    # Mode 14: Children's story
    lambda rng: f"Once upon a time, a tiny drop of water named {rng.choice(_WATER_CHILD_NAMES)} wanted to {rng.choice(_WATER_CHILD_QUESTS)}. So {rng.choice(['Droplet', 'the little drop', 'our hero', 'the brave droplet'])} {rng.choice(['jumped off a cloud', 'flowed down a mountain', 'evaporated into the sky', 'rode a wave to shore'])} and {rng.choice(['saw the whole world below', 'made friends with a fish', 'became part of a rainbow', 'landed in a garden'])}.",
    # Mode 15: JSON/data format
    lambda rng: f'{{"substance": "water", "formula": "H2O", "source": "{rng.choice(_WATER_JSON_SOURCES)}", "quality": "{rng.choice(_WATER_JSON_QUALITY)}", "temperature_c": {rng.choice(["4.0", "15.3", "22.7", "0.1"])}, "ph": {rng.choice(["7.0", "6.8", "7.2", "6.5"])}}}',
]


# ============================================================================
# Scenario 8: Problem-Solving — 15 approach modes, 100 responses
# ============================================================================

_PROBLEM_SOLVING_MAX_MODES: list[Callable[[random.Random], str]] = [
    # Mode 1: Python one-liner
    lambda rng: f"In Python: max_val = max({rng.choice(_MAX_PY_VARS)}). The built-in max() function {rng.choice(_MAX_PY_DOCS)}.",
    # Mode 2: JavaScript function
    lambda rng: (
        lambda v,
        r: f"function findMax({v}) {{\n  let {r} = {v}[0];\n  for (let i = 1; i < {v}.length; i++) {{\n    if ({v}[i] > {r}) {r} = {v}[i];\n  }}\n  return {r};\n}}"
    )(rng.choice(_MAX_JS_VARS), rng.choice(_MAX_JS_RESULT_VARS)),
    # Mode 3: Real-world analogy
    lambda rng: f"Imagine you have a {rng.choice(_MAX_ANALOGY_CONTAINERS)}. Pick up the first one and {rng.choice(_MAX_ANALOGY_ACTIONS)}. Then go through {rng.choice(['each remaining one', 'the rest one by one', 'all the others', 'every single one'])}. If the new one is {rng.choice(['bigger', 'taller', 'heavier', 'larger'])}, {rng.choice(['swap it', 'replace your champion', 'put the old one back', 'update your pick'])}. When you reach the end, {rng.choice(['you are holding the biggest', 'you have your answer', 'the winner is in your hand', 'that is the maximum'])}.",
    # Mode 4: C implementation
    lambda rng: (
        lambda a,
        n,
        r: f"int find_max(int {a}[], int {n}) {{\n    int {r} = {a}[0];\n    for (int i = 1; i < {n}; i++)\n        if ({a}[i] > {r})\n            {r} = {a}[i];\n    return {r};\n}}"
    )(rng.choice(_MAX_C_ARR), rng.choice(_MAX_C_LEN), rng.choice(_MAX_C_RESULT)),
    # Mode 5: Step-by-step instructions
    lambda rng: f"Step 1: {rng.choice(_MAX_STEP1)}.\nStep 2: {rng.choice(_MAX_STEP2)}.\nStep 3: {rng.choice(_MAX_STEP3)}.\nStep 4: {rng.choice(_MAX_STEP4)}.\nThe final value is your answer.",
    # Mode 6: Recursive approach (Python)
    lambda rng: (
        lambda v,
        s: f"def find_max({v}):\n    if len({v}) == 1:\n        return {v}[0]\n    {s} = find_max({v}[1:])\n    return {v}[0] if {v}[0] > {s} else {s}"
    )(rng.choice(_MAX_REC_VARS), rng.choice(_MAX_REC_SUB)),
    # Mode 7: SQL framing
    lambda rng: f"SELECT MAX({rng.choice(_MAX_SQL_COLS)}) FROM {rng.choice(_MAX_SQL_TABLES)};\n\n{rng.choice(_MAX_SQL_EXPLAIN)}. {rng.choice(_MAX_SQL_DETAIL)}.",
    # Mode 8: Divide-and-conquer explanation
    lambda rng: f"Using divide and conquer: {rng.choice(_MAX_DC_SPLIT)}. {rng.choice(_MAX_DC_RECURSE)}. Then {rng.choice(_MAX_DC_MERGE)}. {rng.choice(_MAX_DC_COMPLEXITY)}.",
    # Mode 9: Rust implementation
    lambda rng: (
        lambda a,
        r,
        it: f"fn find_max({a}: &[i32]) -> i32 {{\n    let mut {r} = {a}[0];\n    for &{it} in &{a}[1..] {{\n        if {it} > {r} {{\n            {r} = {it};\n        }}\n    }}\n    {r}\n}}"
    )(
        rng.choice(_MAX_RUST_VARS),
        rng.choice(_MAX_RUST_RESULT),
        rng.choice(_MAX_RUST_ITER),
    ),
    # Mode 10: Mathematical/formal
    lambda rng: f"Given a finite set S = {{x_1, ..., x_n}}, define max(S) = x_i such that x_i >= x_j for all j in {{1,...,n}}. {rng.choice(_MAX_MATH_EXIST)}. {rng.choice(_MAX_MATH_BOUND)}.",
    # Mode 11: Bash/command-line
    lambda rng: f"{rng.choice(_MAX_BASH_CMDS)}\n\n{rng.choice(_MAX_BASH_EXPLAIN)}. {rng.choice(_MAX_BASH_EXTRA)}.",
    # Mode 12: Pros/cons discussion
    lambda rng: f"There are several approaches. A simple {rng.choice(_MAX_PROS_APPROACH)} is O(n) time and O(1) space -- {rng.choice(_MAX_PROS_QUALITY)}. {rng.choice(_MAX_PROS_ALT)}. {rng.choice(_MAX_PROS_ADVICE)}.",
    # Mode 13: Haskell/functional
    lambda rng: (
        lambda v,
        r: f"findMax :: [Int] -> Int\nfindMax [{v}] = {v}\nfindMax ({v}:{r}) = max {v} (findMax {r})\n\n{rng.choice(_MAX_HASKELL_EXPLAIN)}."
    )(rng.choice(_MAX_HASKELL_VARS), rng.choice(_MAX_HASKELL_REST)),
    # Mode 14: Historical/etymology
    lambda rng: f"The problem of finding an extremum dates to {rng.choice(_MAX_HIST_ORIGINS)}. {rng.choice(_MAX_HIST_DETAIL)}. {rng.choice(_MAX_HIST_MODERN)}.",
    # Mode 15: NumPy/vectorized
    lambda rng: (
        lambda v,
        r: f"import numpy as np\n{v} = np.array([{rng.choice(_MAX_NP_ARRAYS)}])\n{r} = np.max({v}){rng.choice(_MAX_NP_COMMENTS)}\n\n{rng.choice(_MAX_NP_EXPLAIN)}."
    )(
        rng.choice(["arr", "data", "values", "numbers"]),
        rng.choice(["result", "max_val", "largest", "answer"]),
    ),
]

_PROBLEM_SOLVING_PALINDROME_MODES: list[Callable[[random.Random], str]] = [
    # Mode 1: Python one-liner
    lambda rng: (
        lambda v: f"In Python: is_palindrome = lambda {v}: {v} == {v}[::-1]. {rng.choice(_PAL_PY_EXPLAIN)}."
    )(rng.choice(_PAL_PY_VARS)),
    # Mode 2: JavaScript function
    lambda rng: (
        lambda v,
        r: f"function isPalindrome({v}) {{\n  const {r} = {v}.split('').reverse().join('');\n  return {v} === {r};\n}}"
    )(rng.choice(_PAL_JS_VARS), rng.choice(_PAL_JS_RESULT)),
    # Mode 3: Real-world analogy
    lambda rng: f"Think of {rng.choice(_PAL_ANALOGY_OBJECTS)}. {rng.choice(_PAL_ANALOGY_METHODS)}. If {rng.choice(_PAL_ANALOGY_CHECKS)} then it is a palindrome. {rng.choice(_PAL_ANALOGY_EXAMPLES)}.",
    # Mode 4: C implementation
    lambda rng: (
        lambda v,
        lo,
        hi: f"int is_palindrome(const char *{v}) {{\n    int {lo} = 0, {hi} = strlen({v}) - 1;\n    while ({lo} < {hi}) {{\n        if ({v}[{lo}++] != {v}[{hi}--])\n            return 0;\n    }}\n    return 1;\n}}"
    )(rng.choice(_PAL_C_VARS), rng.choice(_PAL_C_LEFT), rng.choice(_PAL_C_RIGHT)),
    # Mode 5: Step-by-step instructions
    lambda rng: f"Step 1: {rng.choice(_PAL_STEP1)}.\nStep 2: {rng.choice(_PAL_STEP2)}.\nStep 3: {rng.choice(_PAL_STEP3)}.\nStep 4: {rng.choice(_PAL_STEP4)}.\n{rng.choice(_PAL_STEP5)}.",
    # Mode 6: Recursive approach (Python)
    lambda rng: (
        lambda v: f"def is_palindrome({v}):\n    if len({v}) <= 1:\n        return True\n    if {v}[0] != {v}[-1]:\n        return False\n    return is_palindrome({v}[1:-1])\n\n{rng.choice(_PAL_REC_EXPLAIN)}."
    )(rng.choice(_PAL_PY_VARS)),
    # Mode 7: SQL framing
    lambda rng: (
        lambda v,
        t: f"SELECT {v}\nFROM {t}\nWHERE {v} = REVERSE({v});\n\n{rng.choice(_PAL_SQL_EXPLAIN)}."
    )(rng.choice(_PAL_SQL_VARS), rng.choice(_PAL_SQL_TABLES)),
    # Mode 8: Divide-and-conquer explanation
    lambda rng: f"Using a {rng.choice(_PAL_DC_STRATEGY)} strategy: {rng.choice(_PAL_DC_CHECK)}. If they match, {rng.choice(_PAL_DC_RECURSE)}. {rng.choice(_PAL_DC_ANALYSIS)}.",
    # Mode 9: Rust implementation
    lambda rng: (
        lambda v,
        b,
        lo,
        hi: f"fn is_palindrome({v}: &str) -> bool {{\n    let {b} = {v}.as_bytes();\n    let mut {lo} = 0;\n    let mut {hi} = {b}.len() - 1;\n    while {lo} < {hi} {{\n        if {b}[{lo}] != {b}[{hi}] {{ return false; }}\n        {lo} += 1;\n        {hi} -= 1;\n    }}\n    true\n}}"
    )(
        rng.choice(_PAL_RUST_VARS),
        rng.choice(_PAL_RUST_BYTES),
        rng.choice(_PAL_RUST_LEFT),
        rng.choice(_PAL_RUST_RIGHT),
    ),
    # Mode 10: Mathematical/formal
    lambda rng: f"A string s = s_1 s_2 ... s_n is a palindrome iff s_i = s_{{n+1-i}} for all i in {{1, ..., floor(n/2)}}. {rng.choice(_PAL_MATH1)}. {rng.choice(_PAL_MATH2)}.",
    # Mode 11: Bash/command-line
    lambda rng: f"{rng.choice(_PAL_BASH_CMDS)}\n\n{rng.choice(_PAL_BASH_EXPLAIN)}.",
    # Mode 12: Pros/cons discussion
    lambda rng: f"The {rng.choice(_PAL_PROS_APPROACH)} is {rng.choice(_PAL_PROS_RATING)}. {rng.choice(_PAL_PROS_ALT)}. {rng.choice(_PAL_PROS_ADVICE)}.",
    # Mode 13: Haskell/functional
    lambda rng: (
        lambda t,
        v: f"isPalindrome :: {t} -> Bool\nisPalindrome {v} = {v} == reverse {v}\n\n{rng.choice(_PAL_HASKELL_EXPLAIN)}."
    )(rng.choice(_PAL_HASKELL_TYPES), rng.choice(_PAL_HASKELL_VARS)),
    # Mode 14: Historical/etymology
    lambda rng: f'The word "palindrome" comes from {rng.choice(_PAL_HIST_ORIGIN)}. {rng.choice(_PAL_HIST_EARLY)}. {rng.choice(_PAL_HIST_FAMOUS)}.',
    # Mode 15: NumPy/vectorized
    lambda rng: (
        lambda v: f"import numpy as np\n{v} = np.array(list({rng.choice(_PAL_NP_WORDS)})){rng.choice(_PAL_NP_COMMENTS)}\nis_palindrome = np.all({v} == {v}[::-1])\n\n{rng.choice(_PAL_NP_EXPLAIN)}."
    )(rng.choice(_PAL_NP_VARS)),
]

_PROBLEM_SOLVING_WORDCOUNT_MODES: list[Callable[[random.Random], str]] = [
    # Mode 1: Python one-liner
    lambda rng: f"In Python: count = len({rng.choice(_WC_PY_VARS)}.split()). The split() method {rng.choice(_WC_PY_EXPLAIN)}. {rng.choice(_WC_PY_EXTRA)}.",
    # Mode 2: JavaScript function
    lambda rng: (
        lambda v: f"function countWords({v}) {{\n  return {v}.trim().split(/\\s+/).{rng.choice(_WC_JS_LEN)};\n}}\n\n{rng.choice(_WC_JS_EXPLAIN)}."
    )(rng.choice(_WC_JS_VARS)),
    # Mode 3: Real-world analogy
    lambda rng: f"Imagine you are {rng.choice(_WC_ANALOGY_ACTIONS)}. Every time you {rng.choice(_WC_ANALOGY_TRIGGERS)}, that is a new word. {rng.choice(_WC_ANALOGY_METHODS)}.",
    # Mode 4: C implementation
    lambda rng: (
        lambda v,
        c,
        f: f"int count_words(const char *{v}) {{\n    int {c} = 0, {f} = 0;\n    for (; *{v}; {v}++) {{\n        if (*{v} == ' ' || *{v} == '\\t') {f} = 0;\n        else if (!{f}) {{ {f} = 1; {c}++; }}\n    }}\n    return {c};\n}}"
    )(rng.choice(_WC_C_VARS), rng.choice(_WC_C_COUNT), rng.choice(_WC_C_FLAG)),
    # Mode 5: Step-by-step instructions
    lambda rng: f"Step 1: {rng.choice(_WC_STEP1)}.\nStep 2: {rng.choice(_WC_STEP2)}.\nStep 3: {rng.choice(_WC_STEP3)}.\nStep 4: {rng.choice(_WC_STEP4)}.",
    # Mode 6: Recursive approach (Python)
    lambda rng: (
        lambda v,
        idx: f"def count_words({v}):\n    {v} = {v}.strip()\n    if not {v}:\n        return 0\n    {idx} = {v}.find(' ')\n    if {idx} == -1:\n        return 1\n    return 1 + count_words({v}[{idx} + 1:])\n\n{rng.choice(_WC_REC_EXPLAIN)}."
    )(rng.choice(_WC_REC_VARS), rng.choice(_WC_REC_IDX)),
    # Mode 7: SQL framing
    lambda rng: (
        lambda v,
        t: f"SELECT\n  LENGTH({v}) - LENGTH(REPLACE({v}, ' ', '')) + 1 AS word_count\nFROM {t};\n\n{rng.choice(_WC_SQL_EXPLAIN)}."
    )(rng.choice(_WC_SQL_VARS), rng.choice(_WC_SQL_TABLES)),
    # Mode 8: Divide-and-conquer explanation
    lambda rng: f"Split the sentence {rng.choice(_WC_DC_SPLIT)}. {rng.choice(_WC_DC_RECURSE)}. {rng.choice(_WC_DC_MERGE)}. {rng.choice(_WC_DC_ANALYSIS)}.",
    # Mode 9: Rust implementation
    lambda rng: (
        lambda v: f"fn count_words({v}: &str) -> usize {{\n    {v}.split_whitespace().count()\n}}\n\n{rng.choice(_WC_RUST_EXPLAIN)}."
    )(rng.choice(_WC_RUST_VARS)),
    # Mode 10: Mathematical/formal
    lambda rng: f"Define a word as a maximal contiguous substring of non-whitespace characters. For string s of length n, let f(i) = 1 if s[i] is non-whitespace and (i=0 or s[i-1] is whitespace), else 0. Then word_count(s) = sum_{{i=0}}^{{n-1}} f(i). {rng.choice(_WC_MATH_EXPLAIN)}.",
    # Mode 11: Bash/command-line
    lambda rng: f"{rng.choice(_WC_BASH_CMDS)}\n\n{rng.choice(_WC_BASH_EXPLAIN)}. {rng.choice(_WC_BASH_EXTRA)}.",
    # Mode 12: Pros/cons discussion
    lambda rng: f"The {rng.choice(_WC_PROS_APPROACH)} is {rng.choice(_WC_PROS_RATING)}. {rng.choice(_WC_PROS_ALT)}. {rng.choice(_WC_PROS_ADVICE)}.",
    # Mode 13: Haskell/functional
    lambda rng: f"countWords :: String -> Int\ncountWords = length . words\n\n{rng.choice(_WC_HASKELL_EXPLAIN1)}. {rng.choice(_WC_HASKELL_EXPLAIN2)}.",
    # Mode 14: Historical/etymology
    lambda rng: f"Word counting has been {rng.choice(_WC_HIST)}. {rng.choice(_WC_HIST_DETAIL)}. {rng.choice(_WC_HIST_MODERN)}.",
    # Mode 15: NumPy/vectorized
    lambda rng: (
        lambda v,
        inp,
        sp,
        tr: f"import numpy as np\n{v} = np.frombuffer({inp}.encode(), dtype=np.uint8)\n{sp} = ({v} == ord(' '))\n{tr} = np.diff({sp}.astype(int)) == -1\nword_count = np.sum({tr}) + (1 if not {sp}[0] else 0)\n\n{rng.choice(_WC_NP_EXPLAIN)}."
    )(
        rng.choice(_WC_NP_VARS),
        rng.choice(_WC_NP_INPUT),
        rng.choice(_WC_NP_SPACE),
        rng.choice(_WC_NP_TRANS),
    ),
]


# ============================================================================
# Public constants
# ============================================================================

OPEN_CREATIVE_PROMPTS_AND_RESPONSES: list[tuple[str, list[str]]] = [
    (
        "Write a short piece about rain.",
        _generate_high_diversity_responses(_OPEN_CREATIVE_RAIN_MODES, n=100, seed=70),
    ),
    (
        "Write something about the number seven.",
        _generate_high_diversity_responses(_OPEN_CREATIVE_SEVEN_MODES, n=100, seed=71),
    ),
    (
        "Tell me about water.",
        _generate_high_diversity_responses(_OPEN_CREATIVE_WATER_MODES, n=100, seed=72),
    ),
]
OPEN_CREATIVE_PROMPT_LABELS = ["rain", "seven", "water"]
OPEN_CREATIVE_N_RESPONSES = 100

PROBLEM_SOLVING_PROMPTS_AND_RESPONSES: list[tuple[str, list[str]]] = [
    (
        "How would you find the largest number in a list?",
        _generate_high_diversity_responses(_PROBLEM_SOLVING_MAX_MODES, n=100, seed=80),
    ),
    (
        "How do you check if a word is a palindrome?",
        _generate_high_diversity_responses(
            _PROBLEM_SOLVING_PALINDROME_MODES, n=100, seed=81
        ),
    ),
    (
        "How would you count the number of words in a sentence?",
        _generate_high_diversity_responses(
            _PROBLEM_SOLVING_WORDCOUNT_MODES, n=100, seed=82
        ),
    ),
]
PROBLEM_SOLVING_PROMPT_LABELS = ["find_max", "palindrome", "word_count"]
PROBLEM_SOLVING_N_RESPONSES = 100
