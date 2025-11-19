# AmbedkarGPT RAG Pipeline - Result Analysis

## Evaluation Summary

| Setting | Avg ROUGE-L | Avg Cosine Sim | Avg BLEU | Avg Faithfulness | Avg Relevance |
|---------|-------------|----------------|----------|-----------------|---------------|
| Small   | 0.259       | 0.505          | 0.028    | 0.297           | 0.505         |
| Medium  | 0.264       | 0.532          | 0.039    | 0.375           | 0.532         |
| Large   | 0.269       | 0.521          | 0.05     | 0.417           | 0.521         |

---

## Analysis and Insights

### 1. Which chunking strategy works best for our corpus?
- **Large chunks** yield slightly better ROUGE-L, BLEU, and faithfulness scores compared to Small and Medium chunks.
- This suggests that providing more context per chunk helps the model produce more accurate and faithful answers.
- **Recommendation:** Use Large chunking strategy for this dataset.

### 2. What is our system's current accuracy score?
- Based on relevance and ROUGE-L, the pipeline performs moderately well:
  - Relevance ranges from 0.505 to 0.532
  - ROUGE-L ranges from 0.259 to 0.269
- BLEU scores are still low, indicating that exact token-level matching is weak.
- **Interpretation:** The system captures general meaning but struggles with precise wording.

### 3. What are the most common failure types?
- **Low faithfulness** in some cases: model generates text not directly supported by retrieved documents.
- **Incomplete answers:** Some responses miss specific facts present in the source.
- **Retrieval misses:** Occasionally, the most relevant documents are not retrieved, impacting answer quality.

### 4. What specific improvements would boost performance?
- Improve **document retrieval quality** (better embeddings, more fine-tuned FAISS search).
- Experiment with **overlapping chunks** or dynamic chunk sizes.
- Incorporate **answer grounding** or **source citation** to enhance faithfulness.
- Fine-tune the LLM on **domain-specific corpus** (Ambedkar speeches and writings) to improve BLEU and ROUGE scores.

---

**Conclusion:**  
The RAG pipeline shows promise but needs refinement in retrieval, chunking, and grounding mechanisms to improve faithfulness and overall accuracy. Large chunking currently performs best for this corpus.
