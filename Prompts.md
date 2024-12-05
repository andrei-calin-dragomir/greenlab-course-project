
# Prompt Documentation

## Generation Prompts
### Short Prompt
**Dataset Source**:
Derived from GPT-3 examples in Brown et al., *Few-Shot Learners*, using completion tasks. The style mirrors **OpenAI’s examples** for generating contextual completions.
**Reference Example**:
- GPT-3 paper’s "story continuation" task for few-shot evaluation.
  [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)

### Long Prompt
**Dataset Source**:
Inspired by paragraph continuation tasks in **WikiText-103**. It provides a large-scale dataset for text generation, specifically long-context tasks.
**Reference Example**:
- WikiText examples for narrative continuation tasks.
  [WikiText Dataset](https://paperswithcode.com/dataset/wikitext-103)

## Question-Answering Prompts
### Short Prompt
**Dataset Source**:
Directly inspired by **SQuAD v1.1**, which includes factual Q&A based on short passages.
**Reference Example**:
- SQuAD Question: "What is the capital of Germany?"
  [SQuAD Dataset](https://rajpurkar.github.io/SQuAD-explorer/)

### Long Prompt
**Dataset Source**:
Inspired by multi-paragraph Q&A tasks in **SQuAD v2.0**, which tests contextual reasoning with larger inputs.
**Reference Example**:
- SQuAD Passage: "Greenhouse gases come from human activity such as transportation and industry. The largest contributor is CO2."
  [SQuAD Dataset](https://rajpurkar.github.io/SQuAD-explorer/)

## Summarization Prompts
### Short Prompt
**Dataset Source**:
Derived from **CNN/Daily Mail Dataset**, which contains similar short passages for extractive summarization.
**Reference Example**:
- CNN/Daily Mail Example Article: AI's role in healthcare.
  [CNN/Daily Mail Dataset](https://github.com/abisee/cnn-dailymail)

### Long Prompt
**Dataset Source**:
Based on **CNN/Daily Mail Dataset**, structured for longer summarization tasks with multi-paragraph content.
**Reference Example**:
- CNN/Daily Mail Article Example: Renewable energy challenges and strategies.
  [CNN/Daily Mail Dataset](https://github.com/abisee/cnn-dailymail)


## Derived Input Prompts

The following table describes the input prompts used for each task type:

| **Task Type**          | **Input Type** | **Instruction**                                                                           | **Content**                                                                                                                                                                                                                                                                                                                                                                                                                     | **Expected Output Length**                                                                        |
| ---------------------- | -------------- | ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| **Generation**         | **Short**      | Generate a coherent and contextually appropriate completion for the sentence.             | Artificial intelligence has transformed industries by improving...|100|
|                        | **Long**       | Expand upon the given paragraph with logical, evidence-based details or related concepts. |The Industrial Revolution marked a pivotal moment in human history, with profound impacts on economies, societies, and the environment. One of the lasting consequences of this era is the rise in greenhouse gas emissions, contributing to global warming. Over the years, various international efforts, such as the Kyoto Protocol and the Paris Agreement, have aimed to address this issue. Continuing this discussion, provide a summary of the economic and technological advancements that have emerged as part of the response to climate change.|300|
| **Question Answering** | **Short**      | Provide a precise answer to the following factual question.|What are the capitals of all european countries?|300|
|                        | **Long**       | Analyze the provided context to generate an accurate and well-structured answer.|Climate change is driven by the accumulation of greenhouse gases in the atmosphere, with carbon dioxide being the most significant contributor due to fossil fuel combustion. Other gases like methane and nitrous oxide also play substantial roles. What are the primary sources of these emissions, and how do they vary across different industries?|250|
| **Summarization**      | **Short**      | Summarize the main points from the following brief article.|The adoption of renewable energy sources has been a cornerstone of global strategies to combat climate change. Solar and wind power have seen remarkable growth due to technological advancements and decreasing costs. However, the intermittency of these sources poses a challenge for energy systems, necessitating the development of energy storage technologies and grid integration strategies. Policymakers have implemented incentives, such as tax credits and feed-in tariffs, to accelerate the transition. Nevertheless, achieving carbon neutrality will require a holistic approach, incorporating energy efficiency, sustainable infrastructure development, and international collaboration.|50|
|                        | **Long**       | Provide a concise summary of the key insights from the provided technical paper.|Artificial intelligence (AI), in its broadest sense, is intelligence exhibited by machines, particularly computer systems.  It is a field of research in computer science that develops and studies methods and software that enable machines to perceive their environment and use learning and intelligence to take actions that maximize their chances of achieving defined goals. Such machines may be called AIs. Some high-profile applications of AI include advanced web search engines (e.g., Google Search); recommendation systems (used by YouTube, Amazon, and Netflix); interacting via human speech (e.g., Google Assistant, Siri, and Alexa); autonomous vehicles (e.g., Waymo); generative and creative tools (e.g., ChatGPT, and AI art); and superhuman play and analysis in strategy games (e.g., chess and Go). However, many AI applications are not perceived as AI: A lot of cutting edge AI has filtered into general applications, often without being called AI because once something becomes useful enough and common enough its not labeled AI anymore. The various subfields of AI research are centered around particular goals and the use of particular tools. The traditional goals of AI research include reasoning, knowledge representation, planning, learning, natural language processing, perception, and support for robotics. General intelligence—the ability to complete any task performable by a human on an at least equal level—is among the fields long-term goals. To reach these goals, AI researchers have adapted and integrated a wide range of techniques, including search and mathematical optimization, formal logic, artificial neural networks, and methods based on statistics, operations research, and economics. AI also draws upon psychology, linguistics, philosophy, neuroscience, and other fields. Artificial intelligence was founded as an academic discipline in 1956, and the field went through multiple cycles of optimism, followed by periods of disappointment and loss of funding, known as AI winter. Funding and interest vastly increased after 2012 when deep learning outperformed previous AI techniques. This growth accelerated further after 2017 with the transformer architecture, and by the early 2020s hundreds of billions of dollars were being invested in AI (known as the AI boom). The widespread use of AI in the 21st century exposed several unintended consequences and harms in the present and raised concerns about its risks and long-term effects in the future, prompting discussions about regulatory policies to ensure the safety and benefits of the technology.|150|
