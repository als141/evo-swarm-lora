# GOAL
- docker-compose の vLLM サービスで Multi-LoRA を提供し、OpenAI互換API経由で人格切替を確認。

# ACTIONS
- 起動後の疎通スクリプトを scripts に追加（OpenAI SDK で :persona_x を付与して質問→応答）。
- harness用シェル eval_with_harness.sh を整備。

# ACCEPTANCE
- persona_a/b/c で応答が差別化される。
