# AgentEVAL Prompt Catalog

This directory contains system prompts passed to `DeepSeekJSONClient.complete_json()`.
Dynamic payloads remain in Python code because they depend on snapshots, seeds, cases, and runtime responses.

| File | Used by |
| --- | --- |
| `tool1_seed_review_system.txt` | Tool1 evidence-bound seed review |
| `tool1_semantic_evidence_system.txt` | Tool1 static artifact semantic evidence extraction |
| `tool1_runtime_event_system.txt` | Tool1 runtime response event induction |
| `tool1_siraj_enrichment_system.txt` | Tool1 SIRAJ-style seed metadata enrichment |
| `tool2_variant_system.txt` | Tool2 default LLM setup/trigger text variant |
| `tool2_siraj_case_system.txt` | Tool2 SIRAJ-style case generation |
| `tool2_siraj_refinement_system.txt` | Tool2 SIRAJ-style failed-case refinement |
| `evaluation_direct_llm_baseline_system.txt` | Direct LLM baseline case generation for evaluation |
