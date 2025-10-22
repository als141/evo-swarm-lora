# GOAL
- 3人格の適応度を評価→上位2つを選抜→LoRA交配（alpha_blend）→突然変異→評価→ログ保存のループを構築。

# ACTIONS
- src/coop/fitness.py を用い、タスク性能(正答率or Rouge等)、協調寄与(LOO増分近似)、新規性(埋め込み距離)を合成したfitnessを実装。
- scripts/evolve_loras.py を拡張し、世代ごとに adapters/gen_{t}/... を生成。

# ACCEPTANCE
- N世代回せる（最小2世代）。各世代のfitnessログと最良LoRAの推移が保存。
