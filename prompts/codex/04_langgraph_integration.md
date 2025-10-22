# GOAL
- 現行の自前オーケストレーションを LangGraph v1 に移行（将来、配信や分岐を強化）。

# ACTIONS
- v1の安定APIで、debateノード群（agent→metrics→judge）を状態遷移で表現。
- Qwenローカル推論ラッパーをLangGraphのコール可能に。

# ACCEPTANCE
- 既存run_debate_local.py と同等の結果が LangGraph 実装でも再現。
