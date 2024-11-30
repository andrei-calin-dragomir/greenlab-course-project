
### Generation Prompts
#### Short Prompt
"instruction": "Generate a coherent and contextually appropriate completion for the sentence.",
"content": "Artificial intelligence has transformed industries by improving..."
**Dataset Source**:
Derived from GPT-3 examples in Brown et al., *Few-Shot Learners*, using completion tasks. The style mirrors **OpenAI’s examples** for generating contextual completions.
**Reference Example**:
- GPT-3 paper’s "story continuation" task for few-shot evaluation.
  [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)

#### Long Prompt
"instruction": "Expand upon the given paragraph with logical, evidence-based details or related concepts.",
"content": "The Industrial Revolution marked a pivotal moment in human history, with profound impacts on economies, societies, and the environment. One of the lasting consequences of this era is the rise in greenhouse gas emissions, contributing to global warming. Over the years, various international efforts, such as the Kyoto Protocol and the Paris Agreement, have aimed to address this issue. Continuing this discussion, provide a summary of the economic and technological advancements that have emerged as part of the response to climate change."
**Dataset Source**:
Inspired by paragraph continuation tasks in **WikiText-103**. It provides a large-scale dataset for text generation, specifically long-context tasks.
**Reference Example**:
- WikiText examples for narrative continuation tasks.
  [WikiText Dataset](https://paperswithcode.com/dataset/wikitext-103)

### Question-Answering Prompts
#### Short Prompt
"instruction": "Provide a precise answer to the following factual question.",
"content": "What is the capital of France?",
"expected_output": "Paris"
**Dataset Source**:
Directly inspired by **SQuAD v1.1**, which includes factual Q&A based on short passages.
**Reference Example**:
- SQuAD Question: "What is the capital of Germany?"
  [SQuAD Dataset](https://rajpurkar.github.io/SQuAD-explorer/)

#### Long Prompt
"instruction": "Analyze the provided context to generate an accurate and well-structured answer.",
"content": "Climate change is driven by the accumulation of greenhouse gases in the atmosphere, with carbon dioxide being the most significant contributor due to fossil fuel combustion. Other gases like methane and nitrous oxide also play substantial roles. What are the primary sources of these emissions, and how do they vary across different industries?",
"expected_output": "Primary sources include energy production, agriculture, transportation, and industrial processes, with variations depending on regional practices and technologies."
**Dataset Source**:
Inspired by multi-paragraph Q&A tasks in **SQuAD v2.0**, which tests contextual reasoning with larger inputs.
**Reference Example**:
- SQuAD Passage: "Greenhouse gases come from human activity such as transportation and industry. The largest contributor is CO2."
  [SQuAD Dataset](https://rajpurkar.github.io/SQuAD-explorer/)

### Summarization Prompts
#### Short Prompt
"instruction": "Summarize the main points from the following brief article.",
"content": "The rise of artificial intelligence (AI) in healthcare has opened new frontiers in diagnostics and treatment planning. Machine learning models trained on medical datasets can now predict patient outcomes with unprecedented accuracy. However, challenges remain, including ethical concerns about data privacy, potential biases in AI algorithms, and the need for robust regulatory frameworks. Addressing these issues is crucial for integrating AI into mainstream clinical practice."
**Dataset Source**:
Derived from **CNN/Daily Mail Dataset**, which contains similar short passages for extractive summarization.
**Reference Example**:
- CNN/Daily Mail Example Article: AI's role in healthcare.
  [CNN/Daily Mail Dataset](https://github.com/abisee/cnn-dailymail)

#### Long Prompt
"instruction": "Provide a concise summary of the key insights from the provided technical paper.",
"content": "The adoption of renewable energy sources has been a cornerstone of global strategies to combat climate change. Solar and wind power have seen remarkable growth due to technological advancements and decreasing costs. However, the intermittency of these sources poses a challenge for energy systems, necessitating the development of energy storage technologies and grid integration strategies. Policymakers have implemented incentives, such as tax credits and feed-in tariffs, to accelerate the transition. Nevertheless, achieving carbon neutrality will require a holistic approach, incorporating energy efficiency, sustainable infrastructure development, and international collaboration."
**Dataset Source**:
Based on **CNN/Daily Mail Dataset**, structured for longer summarization tasks with multi-paragraph content.
**Reference Example**:
- CNN/Daily Mail Article Example: Renewable energy challenges and strategies.
  [CNN/Daily Mail Dataset](https://github.com/abisee/cnn-dailymail)
