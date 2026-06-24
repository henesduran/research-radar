<!--
Real output produced by Research Radar's RAG flow, unedited.
Command:
  python ask.py "retrieval augmented generation evaluation" \
                "What metrics or methods are used to evaluate RAG systems?"
The Researcher agent answers using only papers retrieved from the ChromaDB vector
store, with inline citations.
-->

**Question:** What metrics or methods are used to evaluate RAG systems?

---

RAG systems are evaluated using both automated metrics and human-centric methods. Automated frameworks like Ragas and VERA offer quantitative approaches, while industry practices often rely on human evaluation.

**Automated Metrics and Frameworks:**
*   **Ragas (Retrieval Augmented Generation Assessment)** provides a suite of reference-free metrics to evaluate different dimensions of RAG pipelines. These dimensions include the retrieval system's ability to identify relevant and focused context passages, the LLM's capacity to faithfully exploit these passages, and the overall quality of the generation itself ([Ragas: Automated Evaluation of Retrieval Augmented Generation](http://arxiv.org/abs/2309.15217v2)).
*   **VERA (Validation and Evaluation of Retrieval-Augmented Systems)** introduces a cross-encoder-based mechanism that combines multiple multidimensional metrics into a single ranking score. It also utilizes Bootstrap statistics on LLM-based metrics across the document repository to establish confidence bounds ([VERA: Validation and Evaluation of Retrieval-Augmented Systems](http://arxiv.org/abs/2409.03759v1)).

**Human Evaluation:**
*   In industrial contexts, system evaluation for RAG is predominantly conducted by humans rather than automated methods ([Retrieval-Augmented Generation in Industry: An Interview Study on Use Cases, Requirements, Challenges, and Evaluation](http://arxiv.org/abs/2508.14066v1)).

Sources:
*   [Ragas: Automated Evaluation of Retrieval Augmented Generation](http://arxiv.org/abs/2309.15217v2)
*   [Retrieval-Augmented Generation in Industry: An Interview Study on Use Cases, Requirements, Challenges, and Evaluation](http://arxiv.org/abs/2508.14066v1)
*   [VERA: Validation and Evaluation of Retrieval-Augmented Systems](http://arxiv.org/abs/2409.03759v1)
