The ability to give diverse outputs is a necessary component of creativity. 
Prior work uses trained embedding models or surface-level similarities to measure diversity. We realized that we can instead use the in-context learning abilities of a pretrained model to measure how surprising a piece of text remains given other pieces of text. 
This is the a sub n term. The C term adjusts the metric so that incoherent text is considered less diverse.
We validate our approach on both human and AI generations. For human generations, we use Tevet and Berant's benchmarks and find that our approach, which uses no training, approaches but still underperforms trained metrics such as SentBert. For AI generations, we evaluate the OLMO pipeline. A problem with post-training is that it causes mode collapse, and our technique measures that mode collapse as the model moves through the pipeline.


<!-- Creativity researchers and AI evaluators need to measure the diversity of LLM generations and human writing to see how well their elicitation techniques, post-training, or other systems are performing. -->

<!-- In-context learning refers to a pretrained model's ability to identify patterns in text given to it and is the reason why giving LLMs a few examples of a task being done works well at getting them to perform it.  -->
<!-- Our method simply concatenates a prompt with all of responses and measures how surprising the last response is given all previous responses. Averaging over a sample of random permutations eliminates ordering effects, and the measurements are taken per-byte.  -->