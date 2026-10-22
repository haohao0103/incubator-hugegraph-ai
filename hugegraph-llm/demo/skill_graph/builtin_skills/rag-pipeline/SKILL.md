---
name: rag-pipeline
description: Compose retrieval and generation into an end-to-end RAG flow
metadata:
  category: generation
  tags: rag,pipeline
relations:
  depends_on:
    - vector-index
    - reranking
    - prompt-engineering
---

# rag-pipeline

Compose retrieval and generation into an end-to-end RAG flow.

## Steps

1. Understand the inputs and expected outputs.
2. Apply the skill to the current task.
3. Validate the result before handing it off.
