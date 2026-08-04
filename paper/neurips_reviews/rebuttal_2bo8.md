[claude-scaffold: RESPONSE TO REVIEWER 2bo8 (confidence 4, most detailed review). The general response is now posted once under the meta-review; this thread opens with the pointer below instead. Pointer sentence dictated by Matthew; the second sentence is Claude-drafted connective text so the [AI] markers are explained locally — approve or reword. Blocks below are Matthew's verbatim from official_comment.txt lines 20-27.]

Please see our response to the meta-review. As there, any AI contributions are clearly marked between **[AI]** and **[/AI]** markers; content outside them is human-written.

[claude-scaffold: answers 2bo8 W2 + Q2 — judge-model sensitivity]

While it is true that we did not systematically investigate “prompts, n, evaluated-model scale”, we did investigate the judge model, finding that stronger judge models did not improve classification.

**[AI]** Post-submission, we re-scored conTest with judges from three model families spanning 124M to 8B parameters at identical settings: Spearman rho against the human diversity labels is stable across judges (prompt_gen: Qwen2.5-3B 0.58, Llama-3.1-8B 0.57, GPT-2 0.54; resp_gen: 0.39, 0.35, 0.31; story_gen: 0.69, 0.63, 0.71). Pairwise rank agreement between judges is rho 0.76-0.90, and after per-judge standardization, 90-95% of the variance is attributable to the scored item rather than the judge. Furthermore, more recent investigations have shown that when the judges are GPT-2, Llama-3.1-8B, and OLMo-2-7B base (three unrelated model families, none from the pipeline's aligned stages) and the generation models are the four OLMo-2-7B pipeline stages, the generation models account for 86-93% of the variance and judge identity for 5-10%, with the judge-by-generator interaction at only 1-4%. **[/AI]**

This indicates that rankings can be compared across judge models directly, and that absolute values become approximately comparable after a simple per-judge standardization, since the interaction term is the only component a per-judge calibration cannot remove.


We constructed two simple and generic prompts, one for the “instruct” format and one for the “completion” format, and we found them adequate (Appendix B.1). We do not see a particular reason to expect different prompts to perform differently. 

Our perspective is that the choice of ’n’ is determined by the desired sensitivity to tails. If a practitioner wants types of responses with probability greater than 1/m to count as “too common to be diverse”, then the practitioner should set ’n’ to greater than 2m so that that type of response appears more than twice in expectation. For example, if an evaluator wants to know the diversity of the types of responses a model is capable of producing with probability greater than 1/m, they can set n greater than 2m. We will add this guidance to the revised version.

W3: It is true that Decan’s absolute values are difficult to compare across models and meant to be evaluated relative to the same model. This is similarly true of embedding-based metrics like SentBERT. In practice, your concerns are measurable but small: **[AI]** when looking at the Olmo pipeline scored by judges from three unrelated families, 5-10% of the variance is from grader identity and 86-93% from generator identity **[/AI]**.

Appendix C.6 explains why it is not possible to recommend a single metric. There are many ways to make use of in-context learning to measure diversity, and many options are available depending on a practitioner's use case. Appendix B discusses choice of n and recommends fixing it for comparisons.

**[AI]** On W1: we agree the submitted main tables rest on a single judge; the re-scoring above is our direct response, and the revised version will report it alongside the main tables. On W4 and W5: we concede both points; the revised version moves Related Work into the main body and adds the missing citation families, in particular the literature documenting diversity loss in post-trained LLMs. On Q3: the submission contains no instruction-tuned-judge experiments. Post-submission, we re-scored the fixed OLMo generations with the SFT, DPO, and Instruct checkpoints as judges: the stage ordering replicates under every aligned judge, and the confound predicted in Appendix B.1 appears exactly where expected: aligned judges rate base-model text as less coherent (mean C on base-stage text falls from 0.420 under the base judge to 0.331 under the Instruct judge). This supports the base-model recommendation while showing the conclusion is robust to violating it. On Q1, beyond Appendix E's C x E (your reading of its direction is correct), the alternatives we considered are listed in Appendix C.6: weighted sums of a_k - a_inf, slope or curvature at a chosen k, curve-shape descriptors, and coherence-reweighted variants such as C^alpha x a_n; we have not explored that space systematically, as C.6 states. On Q5: example continuations are below. **[/AI]**

<!-- [claude-scaffold: Q5 example block — script-generated by scripts/rlhf_experiment/pick_q5_example.py from results/rlhf_experiment/icl_metrics_length_matched.jsonl and generations_length_matched/{base,instruct}_nbcurated.jsonl; selection is deterministic (median of the NoveltyBench prompts in the top quartile by D drop, not the max). Full file with provenance comment: paper/neurips_reviews/q5_example.md. Response excerpts are verbatim model output (truncated at ~120 chars); the framing text is AI-written.] -->

**[AI]**

**Prompt** (`curated-32`): "I would like you to recommend one graphics card for productivity and light gaming. My budget is $1000 USD."

**D = C x a_n by stage** (this prompt): base 0.534, SFT 0.410, DPO 0.259, RLVR 0.251

**Base** (4 of 10 samples, OLMo-2-7B base):
1. "What's the most powerful CPU to pair? I'll go easy on overclocking and thermals but I need the most power! First, what"
2. "Any tips or links on buying best one? If your is good graphics card Under $1000 : PowerColor R9 270X HD 2GB LY"
3. "Any recommendations?" To decide which graphics card (GPU) is best for your Dell XPS 7018, we need detailed specification"
4. "Thank you very much! The way you talk about failure risk across products makes me wonder, are you bound and determined"

**RLVR** (4 of 10 samples, OLMo-2-7B Instruct/RLVR):
1. "Given your $1000 USD budget, a graphics card that offers good performance for productivity tasks and light gaming is..."
2. "For a productivity and light gaming setup within a $1000 USD budget, I recommend the ASUS ROG Strix RTX 3060 Taylor..."
3. "For a budget of $1,000 USD targeted towards productivity and light gaming, I would recommend the ASUS ROG Strix GT66..."
4. "For a productivity-centric environment with occasional light gaming within a $1,000 USD budget, I would recommend the..."

*Selection rule: among the 189 prompts with metrics at all four stages (base/SFT/DPO/RLVR), we took the top quartile by D(base) - D(RLVR) (48 prompts), restricted to the NoveltyBench subset within it (9 prompts, preferred over AlpacaEval so the example comes from a benchmark built to elicit distinct answers), and picked the median by drop (rank 5 of 9) -- a representative case, not the single largest drop.*

**[/AI]**