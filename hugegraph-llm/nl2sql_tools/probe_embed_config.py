import os, sys
sys.path.insert(0, "hugegraph-llm/src")
from hugegraph_llm.config import llm_settings

for k in ["embedding_type","openai_embedding_model","openai_embedding_api_base",
          "openai_embedding_api_key","openai_embedding_url","ollama_embedding_model",
          "ollama_embedding_host","ollama_embedding_port"]:
    v = getattr(llm_settings, k, "<MISSING>")
    if "KEY" in k.upper() and v:
        v = "<REDACTED len=%d>" % len(v)
    print(k, "=", repr(v))
print("--- env overrides visible to process ---")
for e in sorted(os.environ):
    if "EMBED" in e or "OPENAI_EMBEDDING" in e or "MIHO" in e.upper() or "XIAOMI" in e.upper() or "MIMO" in e.upper():
        print(e, "=", (os.environ[e][:6]+"...") if len(os.environ[e])>8 else os.environ[e])
