"""Format-based mode generators for the mode count experiment.

Each mode is a completely different genre/format (haiku, code, recipe, etc.)
on the topic of "rain". Modes share no structural template with each other —
they are genuinely distinct.

Modes 1–15 reuse the RAIN modes from _new_scenarios.py.
Modes 16–50 add additional distinct formats for higher mode counts.
"""

import random
from typing import Callable

from icl_diversity._new_scenarios import (
    _generate_high_diversity_responses,
    _OPEN_CREATIVE_RAIN_MODES,
)

# ============================================================================
# Additional format modes (16–50) on the topic of "rain"
# ============================================================================

# Mode 16: News article
_NEWS_HEADLINES = (
    "Record Rainfall Hits Eastern Seaboard",
    "Unprecedented Monsoon Season Breaks 50-Year Records",
    "Flash Floods Follow Overnight Downpour",
    "Meteorologists Warn of Extended Rainy Season",
)
_NEWS_DETAILS = (
    "Emergency services reported over 200 calls overnight",
    "Local rivers have risen to dangerously high levels",
    "Schools in three counties have been closed as a precaution",
    "The National Weather Service issued a severe weather advisory",
)

# Mode 17: Tweet/social media
_TWEET_MOODS = (
    "Rainy day mood",
    "This rain tho",
    "Current situation: rain",
    "When it rains it pours fr",
)
_TWEET_TAGS = (
    "#rainyday #cozy #vibes",
    "#rainstorm #weather #mood",
    "#pluviophile #rainlover",
    "#wetweather #stayinside",
)

# Mode 18: Legal disclaimer
_LEGAL_ENTITIES = (
    "RainShield Corp",
    "PluvioTech Industries",
    "AquaGuard LLC",
    "StormSafe International",
)
_LEGAL_CLAUSES = (
    "shall not be held liable for damages caused by rain-related incidents",
    "makes no warranty regarding waterproofing under extreme precipitation",
    "disclaims responsibility for flooding beyond manufacturer specifications",
    "is not responsible for property damage during natural rainfall events",
)

# Mode 19: Product review
_REVIEW_PRODUCTS = (
    "UltraDry Rain Jacket",
    "StormBreaker Umbrella",
    "CloudBurst Rain Boots",
    "RainGuard Windshield Coating",
)
_REVIEW_RATINGS = ("5/5", "4/5", "3/5", "4.5/5")
_REVIEW_VERDICTS = (
    "Kept me completely dry in a torrential downpour",
    "Survived a monsoon but the stitching could be better",
    "Great for light rain, struggles in heavy storms",
    "Best rain gear I have ever owned, worth every penny",
)

# Mode 20: Weather report
_WEATHER_REGIONS = ("the Pacific Northwest", "Southeast Asia", "Northern Europe", "Sub-Saharan Africa")
_WEATHER_AMOUNTS = ("15-25mm", "40-60mm", "5-10mm", "80-120mm")
_WEATHER_OUTLOOKS = (
    "Rain is expected to continue through the weekend",
    "Clearing skies anticipated by Wednesday",
    "Intermittent showers likely for the next 48 hours",
    "A second front may bring additional precipitation midweek",
)

# Mode 21: Interview transcript
_INTERVIEW_EXPERTS = (
    "Dr. Sarah Chen, atmospheric scientist",
    "Prof. James Okafor, hydrologist",
    "Maria Santos, climate researcher",
    "Dr. Anil Gupta, monsoon specialist",
)
_INTERVIEW_QUOTES = (
    "Rain patterns are shifting in ways we did not predict even a decade ago",
    "The relationship between deforestation and local rainfall is stronger than most realize",
    "We are seeing rainfall events that were once-in-a-century becoming annual occurrences",
    "Understanding rain at the molecular level has transformed weather prediction",
)

# Mode 22: Classified ad
_AD_ITEMS = (
    "RAIN BARREL - 55 gallon, food-grade plastic",
    "VINTAGE UMBRELLA COLLECTION - 12 pieces, various eras",
    "RAIN GAUGE SET - professional grade, stainless steel",
    "WATERPROOF CAMPING GEAR - tent + tarp, barely used",
)
_AD_PRICES = ("$45 OBO", "$120 firm", "$30", "$200 or best offer")
_AD_CONTACTS = (
    "Call after 5pm, ask for Mike",
    "Text preferred, serious buyers only",
    "Email rainlover99@mail.com",
    "Pick up in Rainyville, no delivery",
)

# Mode 23: Academic abstract
_ABSTRACT_TITLES = (
    "Spatiotemporal Variability in Tropical Rainfall Intensity",
    "Microphysical Properties of Raindrops in Convective Systems",
    "Urban Heat Island Effects on Local Precipitation Patterns",
    "Statistical Modeling of Extreme Rainfall Return Periods",
)
_ABSTRACT_FINDINGS = (
    "Our results indicate a 12% increase in extreme rainfall events per decade",
    "We find that drop size distributions follow a modified gamma function",
    "Analysis reveals significant urban-rural precipitation gradients",
    "The fitted GPD model outperforms traditional GEV approaches for tail estimation",
)

# Mode 24: Instruction manual
_MANUAL_DEVICES = (
    "Rain Sensor Model RS-3000",
    "Automated Rain Gauge AG-500",
    "PluvioMatic Digital Rain Collector",
    "WeatherStation Pro Rain Module",
)
_MANUAL_STEPS = (
    "Mount the sensor 1 meter above ground on a level surface",
    "Connect the USB cable to your data logger",
    "Calibrate by pouring exactly 100ml of water into the funnel",
    "Ensure the tipping bucket moves freely before deployment",
)

# Mode 25: Movie script scene
_SCRIPT_SETTINGS = (
    "EXT. CITY STREET - NIGHT - RAIN",
    "EXT. COUNTRY ROAD - DAWN - HEAVY RAIN",
    "EXT. ROOFTOP - EVENING - DRIZZLE",
    "EXT. BEACH - AFTERNOON - SUDDEN DOWNPOUR",
)
_SCRIPT_ACTIONS = (
    "A figure in a dark coat steps into the downpour, face tilted upward",
    "Two children run laughing through puddles as lightning flashes",
    "She stands at the edge, letting the rain wash away the mascara",
    "He holds a newspaper over his head, futile against the deluge",
)

# Mode 26: Encyclopedia entry
_ENCY_OPENINGS = (
    "Rain, also known as precipitation in liquid form",
    "Rainfall is a component of the water cycle",
    "Rain is the condensation of atmospheric water vapor",
    "Precipitation in the form of liquid water droplets",
)
_ENCY_FACTS = (
    "is classified by intensity as light (<2.5mm/h), moderate (2.5-7.5mm/h), or heavy (>7.5mm/h)",
    "accounts for the primary mechanism of freshwater delivery to terrestrial ecosystems",
    "forms through either the Bergeron process or the collision-coalescence mechanism",
    "varies dramatically by latitude, altitude, and proximity to moisture sources",
)

# Mode 27: Crossword clue
_CLUE_STYLES = (
    "Precipitation from clouds (4) — RAIN",
    "Wet weather, rhymes with 'pain' (4) — RAIN",
    "What an umbrella protects against (4) — RAIN",
    "Downpour; liquid from the sky (4) — RAIN",
)
_CLUE_BONUS = (
    "Related: DRIZZLE (7), SHOWER (6), MONSOON (7), DELUGE (6)",
    "See also: 14-Across 'UMBRELLA', 23-Down 'PUDDLE'",
    "Anagram of IRAN; also a name (Lorraine)",
    "Homophone: REIGN, REIN. Triple homophone!",
)

# Mode 28: Complaint letter
_COMPLAINT_ISSUES = (
    "the persistent leaking in my apartment ceiling during every rainstorm",
    "the inadequate drainage system on Elm Street that floods with minimal rain",
    "the failure of your roofing warranty to cover rain damage",
    "the constant puddles forming in your parking lot after any rainfall",
)
_COMPLAINT_DEMANDS = (
    "I expect repairs to be completed within 14 business days",
    "Please send a qualified inspector within the week",
    "I am requesting a full refund under the warranty terms",
    "Immediate action is required before the next storm season",
)

# Mode 29: Riddle
_RIDDLE_TEXTS = (
    "I fall but never break. I am clear but not glass. I come from above but I am not a bird. What am I? Rain.",
    "I have no legs but I run down hills. I have no mouth but I make the river full. I am tiny alone but powerful together. The answer is rain.",
    "Born in a cloud, I die on the ground. I make flowers happy and basements frown. What am I? Rain.",
    "You can hear me but not see me coming. You can feel me but not hold me. I feed the rivers and fill the wells. I am rain.",
)

# Mode 30: Court transcript
_COURT_CASES = (
    "In the matter of Henderson v. City of Portland, regarding storm drainage",
    "Case No. 2024-CV-4521: Plaintiff alleges negligent rainwater management",
    "Testimony of expert witness Dr. Plume regarding rainfall on the night in question",
    "The State v. AquaCorp Inc., environmental rainfall contamination hearing",
)
_COURT_TESTIMONY = (
    "The rainfall intensity exceeded design capacity by approximately 40 percent",
    "Records show 78mm of rain fell in a two-hour window, well above the 50-year threshold",
    "The defendant failed to maintain drainage infrastructure as required by municipal code",
    "Meteorological data confirms the rainfall was within normal seasonal parameters",
)

# Mode 31: Recipe review
_RECIPE_REVIEW_DISHES = (
    "Rainy Day Chicken Soup",
    "Stormy Weather Chocolate Cake",
    "Monsoon Chai Latte",
    "Drizzle Honey Scones",
)
_RECIPE_REVIEW_OPINIONS = (
    "Perfect comfort food when the rain is coming down. My family devoured it",
    "Made this during a thunderstorm and it was exactly what we needed",
    "The recipe calls for too much salt but otherwise great for a rainy afternoon",
    "Five stars. I make this every time it rains now. So warm and satisfying",
)

# Mode 32: Sports commentary
_SPORTS_EVENTS = (
    "The match has been delayed by 45 minutes due to persistent rain",
    "Rain is making the pitch incredibly slippery here at Wembley",
    "Conditions are deteriorating as the rain intensifies in the third quarter",
    "The outfield is soaked and the umpires are inspecting the crease",
)
_SPORTS_IMPACT = (
    "Both teams are struggling with ball control on the wet surface",
    "The wet track has completely changed the tire strategy calculus",
    "Visibility is poor and the crowd is thinning under the downpour",
    "This rain could be the deciding factor in today's championship",
)

# Mode 33: Wiki infobox style
_WIKI_TYPES = ("Meteorological phenomenon", "Atmospheric precipitation", "Hydrological event", "Weather type")
_WIKI_RELATED = (
    "Drizzle, Sleet, Snow, Hail",
    "Monsoon, Typhoon, Cyclone",
    "Fog, Mist, Dew, Frost",
    "Thunderstorm, Squall, Cloudburst",
)

# Mode 34: Fortune cookie
_FORTUNE_TEXTS = (
    "After the rain comes the rainbow. Your patience will be rewarded soon.",
    "Like rain on a tin roof, small joys make the sweetest music.",
    "Even rain must fall to help flowers grow. Embrace the storms ahead.",
    "The rain does not ask permission. Neither should your dreams.",
)

# Mode 35: Horoscope
_HOROSCOPE_SIGNS = ("Aquarius", "Pisces", "Cancer", "Scorpio")
_HOROSCOPE_READINGS = (
    "A rainy day brings unexpected clarity to a long-standing problem",
    "The sound of rain will wash away your doubts this week",
    "Rain on your birthday signals a year of emotional renewal",
    "A conversation during a rainstorm will change your perspective",
)

# Mode 36: Error message
_ERROR_CODES = ("RAIN_OVERFLOW_ERR", "PRECIP_TIMEOUT", "CLOUD_BUFFER_FULL", "DRAIN_CAPACITY_EXCEEDED")
_ERROR_DETAILS = (
    "Maximum rainfall rate exceeded. Current: 120mm/h. Limit: 75mm/h",
    "Rain event duration exceeded timeout of 72 hours. Process terminated",
    "Cumulus buffer at 100% capacity. Cannot allocate additional moisture",
    "Storm drain throughput saturated. Backpressure detected at node 14",
)

# Mode 37: Changelog
_CHANGELOG_VERSIONS = ("v2.4.0", "v3.1.0", "v1.8.2", "v4.0.0-beta")
_CHANGELOG_ENTRIES = (
    "Added support for tropical rainfall patterns in Southern Hemisphere",
    "Fixed rain intensity calculation overflow for extreme events (>200mm/h)",
    "Improved rain/snow boundary detection at temperatures near 0C",
    "Breaking: Redesigned precipitation API to support real-time streaming",
)

# Mode 38: Haiku variant (senryu — human-focused)
_SENRYU_TEXTS = (
    "Forgot my umbrella.\nThe bus is three minutes late.\nOf course it's raining.",
    "She watches the rain.\nHer coffee grows cold, untouched.\nMemories pour down.",
    "Monday morning rain.\nThe alarm did not go off.\nNature says stay home.",
    "Children in raincoats.\nJumping in every puddle.\nParents pretending not to want to join.",
)

# Mode 39: Limerick
_LIMERICK_TEXTS = (
    "A raindrop that fell on a cat\nSlid right off and landed ker-splat.\nThe cat looked around,\nAt the wet on the ground,\nAnd decided to go find a mat.",
    "There once was some rain in Spokane,\nThat fell on a man and his cane.\nHe slipped on the street,\nLanded right on his feet,\nAnd declared he would not walk again.",
    "A cloud over London one day\nDecided to rain and to stay.\nThe Brits grabbed their tea,\nSaid 'Quite right, let it be,'\nAnd went on about their normal way.",
    "The rain came to visit the shore,\nAnd knocked on each cottage's door.\nThe sea said 'Come in!'\nWith a watery grin,\nAnd together they made quite a pour.",
)

# Mode 40: Acrostic
_ACROSTIC_TEXTS = (
    "Rivulets running down the windowpane,\nA world washed clean and new again,\nIn every drop a tiny mirror,\nNature's voice, now soft, now clearer.",
    "Rushing from clouds to the ground below,\nA billion droplets in the flow,\nImpossibly light yet shaping stone,\nNever the same, yet always known.",
    "Rooftops drum with silver sound,\nAncient water, skyward bound,\nInfinite cycle, round and round,\nNourishing every plot of ground.",
    "Rivers start with just one drop,\nA gentle fall that will not stop,\nInto streams and lakes it goes,\nNobody knows where the first rain chose.",
)

# Mode 41: Lab notebook
_LAB_DATES = ("2024-03-15", "2024-07-22", "2024-11-03", "2024-09-18")
_LAB_OBSERVATIONS = (
    "Collected 47ml of rainwater from rooftop station. pH measured at 5.6 (slightly acidic, typical). Conductivity: 23 uS/cm",
    "Rain event began 14:32, ended 16:05. Total accumulation: 18.3mm. Drop size distribution peaked at 1.8mm",
    "Filtered sample through 0.45um membrane. Residue mass: 2.1mg/L. Ion chromatography pending",
    "Compared rain gauge readings: tipping bucket 22.1mm vs weighing gauge 22.4mm. Within calibration tolerance",
)

# Mode 42: Field notes
_FIELD_LOCATIONS = ("Borneo rainforest transect B7", "Sahel rainfall monitoring station", "Pacific NW old-growth site", "Atacama fog-collection array")
_FIELD_NOTES = (
    "Rain began at 0630. Canopy interception estimated at 35%. Understory throughfall patchy. Soil moisture sensors responding within 20 min",
    "First significant rain in 47 days. Community gathering at collection point. Millet seedlings showing immediate turgor response",
    "Moss samples saturated within first 10mm. Epiphyte drip contributing to stemflow. Spotted varied thrush sheltering under cedar",
    "Fog event produced 3.2L/m2 on mesh collectors. Not technically rain but functionally equivalent for local ecology",
)

# Mode 43: Ship's log
_SHIP_DATES = ("Day 14, 0800 hours", "Stardate 47.3, morning watch", "March 15th, dog watch", "Day 7, first bell")
_SHIP_ENTRIES = (
    "Heavy rain since midnight. Visibility reduced to 200 meters. Crew morale holding. Adjusted heading 10 degrees south to skirt the worst of it",
    "Squall line passed over at dawn. Collected 80 gallons of freshwater from the sails. A welcome reprieve from rationing",
    "Rain and wind from the northeast. Barometer falling steadily. Secured all hatches. The cook managed a hot meal despite the roll",
    "Light rain clearing by noon. Used the opportunity to wash salt from the rigging. The navigator reports we are 3 days from port",
)

# Mode 44: Telegram
_TELEGRAM_TEXTS = (
    "HEAVY RAIN STOP ROADS IMPASSABLE STOP SEND SUPPLIES BY AIR STOP SITUATION MANAGEABLE STOP",
    "RAIN DELAY THREE DAYS STOP CONSTRUCTION HALTED STOP COST OVERRUN ESTIMATED STOP ADVISE STOP",
    "MONSOON ARRIVED EARLY STOP CROPS FLOODING STOP REQUEST EMERGENCY DRAINAGE PUMPS STOP",
    "ARRIVED SAFELY DESPITE RAIN STOP UMBRELLA LOST IN WIND STOP SPIRITS HIGH STOP LOVE STOP",
)

# Mode 45: Postcard
_POSTCARD_TEXTS = (
    "Dear Mom, It has rained every day since we arrived in London but we love it. The museums are wonderful when it pours. Bought you a rain hat at Camden Market. Miss you! -J",
    "Greetings from Seattle! They were not kidding about the rain. But the coffee shops are so cozy when it is wet outside. Having a wonderful time despite soggy shoes. xo",
    "Hello from Cherrapunji! The locals say this is a 'light' rainy season. I have never seen so much water fall from the sky. The waterfalls are magnificent. Wish you were here.",
    "Writing this from a cafe in Paris while it pours outside. The Seine is high and the bridges gleam. Rain makes this city even more beautiful somehow. A bientot!",
)

# Mode 46: Prayer/blessing
_PRAYER_TEXTS = (
    "We give thanks for the rain that nourishes our fields and fills our wells. May this water sustain us through the dry months ahead.",
    "Blessed is the rain that falls on the just and the unjust alike. May we receive this gift with gratitude and use it wisely.",
    "O gentle rain, wash the dust from our weary world. Bring new life to the seeds sleeping in the soil. We welcome your return.",
    "For the farmers waiting, for the rivers running low, for the earth that thirsts — may the rains come in their season, neither too early nor too late.",
)

# Mode 47: Toast/speech
_TOAST_TEXTS = (
    "Ladies and gentlemen, they said it would rain on our wedding day. And it did! But as my grandmother always said, rain on your wedding day means a long and fruitful marriage. So raise your glasses — to love, to laughter, and to dancing in the rain!",
    "I would like to propose a toast. To the rain that cancelled our outdoor plans and brought us all together in this tiny kitchen. Some of the best conversations happen when you cannot go anywhere. Cheers!",
    "Here is to the rain — the great equalizer. It does not care if you are rich or poor, happy or sad. It falls on everyone. And somehow, shared rain brings people closer. To shared umbrellas and new friendships!",
    "They say into every life a little rain must fall. Well, this year we got more than a little. But we are still here, still standing, still laughing. So here is to weathering every storm together. Salud!",
)

# Mode 48: Eulogy
_EULOGY_TEXTS = (
    "She always loved the rain. Even as a girl, she would stand at the window and watch it for hours. She said each drop was a tiny traveler with a story to tell. Today it is raining, and I like to think she arranged it.",
    "Dad used to say that rain was the sky doing its laundry. He had a joke for everything, even the weather. He built us a rain gauge when we were kids and made us record every storm. I still have the notebook.",
    "Grandpa was a farmer, and to him, rain was not weather — it was livelihood. He could tell you by the smell of the air whether rain was coming. He was right more often than the forecast. The fields are quiet today.",
    "She danced in the rain the day she retired. No umbrella, no coat, just pure joy. That is how I want to remember her. Unafraid, joyful, and always ready to get a little wet for something wonderful.",
)

# Mode 49: Cover letter
_COVER_OPENINGS = (
    "I am writing to apply for the position of Rainfall Data Analyst",
    "Please consider my application for the Precipitation Research Fellowship",
    "I am excited to apply for the role of Storm Systems Engineer",
    "With five years of experience in atmospheric science, I am applying for the Rain Monitoring Specialist position",
)
_COVER_SKILLS = (
    "My expertise in radar rainfall estimation and ground-truth validation makes me an ideal candidate",
    "I have published three papers on tropical rainfall variability and climate feedback loops",
    "I developed the rain prediction algorithm currently used by two national weather services",
    "My dissertation on monsoon dynamics was awarded the AMS Outstanding Thesis Prize",
)

# Mode 50: Thesis statement
_THESIS_TEXTS = (
    "This thesis argues that changes in global rainfall patterns over the past century are primarily driven by anthropogenic aerosol emissions rather than greenhouse gas forcing alone, as demonstrated through analysis of 847 rain gauge stations across six continents.",
    "I contend that the cultural significance of rain in literature has shifted from a symbol of divine judgment in pre-modern texts to a metaphor for psychological renewal in contemporary fiction, as evidenced by computational analysis of 12,000 English-language novels.",
    "This dissertation demonstrates that machine learning models trained on disdrometer data can predict raindrop size distributions with 94% accuracy, outperforming traditional Marshall-Palmer parameterizations by a factor of three.",
    "The central argument of this work is that decentralized rainwater harvesting systems are more resilient and cost-effective than centralized infrastructure for communities in semi-arid regions, supported by case studies from 15 communities across three continents.",
)


# ============================================================================
# Mode list builders
# ============================================================================

def _build_additional_modes() -> list[Callable[[random.Random], str]]:
    """Build modes 16–50 as callables matching the (rng) -> str interface."""
    return [
        # 16: News article
        lambda rng: f"BREAKING: {rng.choice(_NEWS_HEADLINES)}. {rng.choice(_NEWS_DETAILS)}. More details at 11.",
        # 17: Tweet/social media
        lambda rng: f"{rng.choice(_TWEET_MOODS)} ☔ {rng.choice(_TWEET_TAGS)}",
        # 18: Legal disclaimer
        lambda rng: f"NOTICE: {rng.choice(_LEGAL_ENTITIES)} {rng.choice(_LEGAL_CLAUSES)}. By proceeding, you acknowledge these terms.",
        # 19: Product review
        lambda rng: f"{rng.choice(_REVIEW_PRODUCTS)} — {rng.choice(_REVIEW_RATINGS)} stars. {rng.choice(_REVIEW_VERDICTS)}.",
        # 20: Weather report
        lambda rng: f"Weather Advisory for {rng.choice(_WEATHER_REGIONS)}: Expected rainfall of {rng.choice(_WEATHER_AMOUNTS)} over the next 24 hours. {rng.choice(_WEATHER_OUTLOOKS)}.",
        # 21: Interview transcript
        lambda rng: f"Q: What is the most surprising thing about rain?\n{rng.choice(_INTERVIEW_EXPERTS)}: \"{rng.choice(_INTERVIEW_QUOTES)}.\"",
        # 22: Classified ad
        lambda rng: f"FOR SALE: {rng.choice(_AD_ITEMS)}. {rng.choice(_AD_PRICES)}. {rng.choice(_AD_CONTACTS)}.",
        # 23: Academic abstract
        lambda rng: f"Title: {rng.choice(_ABSTRACT_TITLES)}. Abstract: {rng.choice(_ABSTRACT_FINDINGS)}. Keywords: precipitation, rainfall, climate.",
        # 24: Instruction manual
        lambda rng: f"{rng.choice(_MANUAL_DEVICES)} — Quick Start Guide. Step 1: {rng.choice(_MANUAL_STEPS)}. Refer to full manual for troubleshooting.",
        # 25: Movie script scene
        lambda rng: f"{rng.choice(_SCRIPT_SETTINGS)}\n\n{rng.choice(_SCRIPT_ACTIONS)}.",
        # 26: Encyclopedia entry
        lambda rng: f"{rng.choice(_ENCY_OPENINGS)}, {rng.choice(_ENCY_FACTS)}.",
        # 27: Crossword clue
        lambda rng: f"{rng.choice(_CLUE_STYLES)}. {rng.choice(_CLUE_BONUS)}.",
        # 28: Complaint letter
        lambda rng: f"To whom it may concern: I am writing to complain about {rng.choice(_COMPLAINT_ISSUES)}. {rng.choice(_COMPLAINT_DEMANDS)}.",
        # 29: Riddle
        lambda rng: rng.choice(_RIDDLE_TEXTS),
        # 30: Court transcript
        lambda rng: f"{rng.choice(_COURT_CASES)}. Witness states: \"{rng.choice(_COURT_TESTIMONY)}.\"",
        # 31: Recipe review
        lambda rng: f"Review of {rng.choice(_RECIPE_REVIEW_DISHES)}: {rng.choice(_RECIPE_REVIEW_OPINIONS)}.",
        # 32: Sports commentary
        lambda rng: f"{rng.choice(_SPORTS_EVENTS)}. {rng.choice(_SPORTS_IMPACT)}.",
        # 33: Wiki infobox
        lambda rng: f"Rain | Type: {rng.choice(_WIKI_TYPES)} | Related: {rng.choice(_WIKI_RELATED)} | Measurement: mm, in | Frequency: Global",
        # 34: Fortune cookie
        lambda rng: rng.choice(_FORTUNE_TEXTS),
        # 35: Horoscope
        lambda rng: f"{rng.choice(_HOROSCOPE_SIGNS)} weekly forecast: {rng.choice(_HOROSCOPE_READINGS)}.",
        # 36: Error message
        lambda rng: f"[{rng.choice(_ERROR_CODES)}] {rng.choice(_ERROR_DETAILS)}. Contact sysadmin.",
        # 37: Changelog
        lambda rng: f"## {rng.choice(_CHANGELOG_VERSIONS)}\n- {rng.choice(_CHANGELOG_ENTRIES)}.",
        # 38: Senryu
        lambda rng: rng.choice(_SENRYU_TEXTS),
        # 39: Limerick
        lambda rng: rng.choice(_LIMERICK_TEXTS),
        # 40: Acrostic
        lambda rng: rng.choice(_ACROSTIC_TEXTS),
        # 41: Lab notebook
        lambda rng: f"Date: {rng.choice(_LAB_DATES)}. Obs: {rng.choice(_LAB_OBSERVATIONS)}.",
        # 42: Field notes
        lambda rng: f"Site: {rng.choice(_FIELD_LOCATIONS)}. {rng.choice(_FIELD_NOTES)}.",
        # 43: Ship's log
        lambda rng: f"{rng.choice(_SHIP_DATES)}. {rng.choice(_SHIP_ENTRIES)}.",
        # 44: Telegram
        lambda rng: rng.choice(_TELEGRAM_TEXTS),
        # 45: Postcard
        lambda rng: rng.choice(_POSTCARD_TEXTS),
        # 46: Prayer/blessing
        lambda rng: rng.choice(_PRAYER_TEXTS),
        # 47: Toast/speech
        lambda rng: rng.choice(_TOAST_TEXTS),
        # 48: Eulogy
        lambda rng: rng.choice(_EULOGY_TEXTS),
        # 49: Cover letter
        lambda rng: f"{rng.choice(_COVER_OPENINGS)}. {rng.choice(_COVER_SKILLS)}.",
        # 50: Thesis statement
        lambda rng: rng.choice(_THESIS_TEXTS),
    ]


# All 50 modes: 15 from _new_scenarios + 35 additional
_ALL_RAIN_MODES: list[Callable[[random.Random], str]] = (
    list(_OPEN_CREATIVE_RAIN_MODES) + _build_additional_modes()
)

# Mode names for labeling
MODE_NAMES: list[str] = [
    "haiku", "python_code", "recipe", "scientific_fact", "diary",
    "numbered_list", "dialogue", "philosophy", "song_lyrics", "math_stats",
    "metaphor", "historical_fact", "letter", "childrens_story", "json_data",
    "news_article", "tweet", "legal_disclaimer", "product_review", "weather_report",
    "interview", "classified_ad", "academic_abstract", "instruction_manual", "movie_script",
    "encyclopedia", "crossword_clue", "complaint_letter", "riddle", "court_transcript",
    "recipe_review", "sports_commentary", "wiki_infobox", "fortune_cookie", "horoscope",
    "error_message", "changelog", "senryu", "limerick", "acrostic",
    "lab_notebook", "field_notes", "ships_log", "telegram", "postcard",
    "prayer", "toast_speech", "eulogy", "cover_letter", "thesis_statement",
]

PROMPT = "Write a short piece about rain."


def get_format_modes(m: int) -> list[Callable[[random.Random], str]]:
    """Return m format modes for the rain topic.

    Args:
        m: Number of modes to return (1–50).

    Returns:
        List of m mode callables, each accepting a random.Random and returning a string.

    Raises:
        ValueError: If m < 1 or m > 50.
    """
    if m < 1 or m > len(_ALL_RAIN_MODES):
        raise ValueError(f"m must be between 1 and {len(_ALL_RAIN_MODES)}, got {m}")
    return _ALL_RAIN_MODES[:m]


def generate_mode_count_responses(
    m: int,
    n_per_mode: int = 4,
    seed: int = 0,
) -> list[str]:
    """Generate responses using m format modes.

    Uses _generate_high_diversity_responses with exactly m modes,
    producing n_per_mode responses per mode (total = m * n_per_mode).

    Args:
        m: Number of distinct modes.
        n_per_mode: Responses per mode.
        seed: Random seed.

    Returns:
        List of m * n_per_mode responses.
    """
    modes = get_format_modes(m)
    return _generate_high_diversity_responses(modes, n=m * n_per_mode, seed=seed)
