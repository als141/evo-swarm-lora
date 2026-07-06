# -*- coding: utf-8 -*-
"""研究進捗スライド（2026-07-09 報告）を組み立てる。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from deck_helpers import *  # noqa

HERE = Path(__file__).parent
AST = HERE / "assets"
prs = new_deck()

TOTAL = 26          # 本編ページ数（章扉・表紙・付録は番号を振らない）
_page = {"n": 0}


def foot(slide):
    _page["n"] += 1
    footer(slide, _page["n"], TOTAL)


def apx_foot(slide, idx, total=3):
    text(slide, 0.85, 7.06, 8.0, 0.3,
         [("LoRAエージェント集団の進化 ／ 研究進捗", 9.5, FAINT, FONT_L, {})],
         anchor=MSO_ANCHOR.MIDDLE)
    text(slide, 11.2, 7.06, 1.3, 0.3, [(f"付録 A{idx} / {total}", 9.5, FAINT, FONT_L, {})],
         align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE)


# ============================================================
# 表紙
# ============================================================
def cover():
    s = add_slide(prs, bg=PAPER)
    rect(s, 0, 0, 13.333, 0.28, fill=INK_PANEL)
    rect(s, 0, 7.22, 13.333, 0.28, fill=ACCENT)
    text(s, 0.95, 1.15, 11.4, 0.5,
         [("修士研究 進捗報告", 15, ACCENT, FONT_SB, {})])
    text(s, 0.9, 1.75, 11.6, 2.2, [
        ("小さなAIを3体、話し合わせて“進化”させる", 33, INK, FONT, {"bold": True}),
    ], line_spacing=1.18)
    text(s, 0.92, 2.62, 11.6, 1.4, [
        ("——「議論するAIチーム」は、素朴なやり方に勝てるのか？", 19, SUB, FONT, {}),
    ])
    # 英題
    text(s, 0.95, 3.7, 11.5, 0.9, [
        ("Evolutionary Optimization of LoRA Agent Populations", 14, FAINT, FONT_L, {}),
        ("\nwith Team-Level Fitness in Multi-Agent Debate", 14, FAINT, FONT_L, {}),
    ], line_spacing=1.2)
    line(s, 0.95, 5.25, 5.3, 0, color=LINE, weight=1.3)
    para_block(s, 0.95, 5.5, 11.0, 1.4, [
        [("新潟大学 自然科学研究科　電気情報工学専攻　情報社会デザイン科学コース", 14, INK, FONT, {})],
        [("舛田 岳　（学籍番号 F25C142E）", 15, INK, FONT, {"bold": True})],
        [("2026年7月9日", 13, SUB, FONT_L, {})],
    ], space_after=6)
    notes(s, "本日はよろしくお願いします。修士研究の進捗を報告します。"
             "テーマは、小さなAIを3体用意して話し合わせ、そのチームを進化のしくみで良くしていく、というものです。"
             "今日いちばん見ていただきたいのは、実際にAIたちがどう会話しているか、その生のログです。"
             "結論から言うと、正直に良かった点とダメだった点の両方が出ました。そこも包み隠さずお話しします。")


# ============================================================
# 1. ワンライナー / 今日見せるもの
# ============================================================
def sl_intro():
    s = add_slide(prs)
    title_head(s, "この研究を一言でいうと", "「1体の賢いAI」より「3体で話し合うAI」は強いのか？")
    # 3枚のカード
    cards = [
        ("やったこと", TEAL, [
            "同じ小型AIに3つの“性格”を持たせる",
            "互いの答えを見せ合って議論させる",
            "良い協調をする個体を“進化”で選ぶ",
        ]),
        ("調べたこと", ACCENT, [
            "議論チームは、単純な多数決に勝てるか",
            "どんな問題なら議論が効く／効かないか",
            "うまくいかない原因はどこにあるか",
        ]),
        ("今日の見どころ", POS, [
            "本物の会話ログでAIの議論を実演",
            "成功する例と、崩れる例の両方",
            "正直な結果と、その理由の分析",
        ]),
    ]
    x0, w, gap = 0.85, 3.72, 0.19
    for i, (hd, col, items) in enumerate(cards):
        x = x0 + i * (w + gap)
        rect(s, x, 2.15, w, 3.7, fill=CARD, line=LINE, line_w=1.1,
             shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.045, shadow=True)
        rect(s, x, 2.15, w, 0.62, fill=col, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.045)
        rect(s, x, 2.45, w, 0.32, fill=col)  # 角丸の下を埋める
        text(s, x, 2.15, w, 0.62, [(hd, 16, CARD, FONT_SB, {})],
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        bullets(s, x + 0.32, 3.02, w - 0.6, 2.7, items, size=13.5, gap=12,
                marker="●", marker_color=col)
    text(s, 0.85, 6.15, 11.6, 0.7,
         [("キーワードは ", 15, INK, FONT, {}),
          ("「多様性」と「協調」", 15, ACCENT, FONT, {"bold": True}),
          ("。ただし、いいことばかりではありませんでした。", 15, INK, FONT, {})])
    foot(s)
    notes(s, "まず全体像です。この研究でやったのは大きく3つ。"
             "1つの土台モデルに3つの性格を持たせ、互いの答えを見せ合って議論させ、"
             "協調がうまい個体を進化で選ぶ、というものです。"
             "調べたのは、こうした議論チームが、単純に何回も解いて多数決するやり方に勝てるのか。"
             "どんな問題で効いて、どんな問題でダメなのか。そしてダメなときの原因はどこか。"
             "今日は本物の会話ログを使って、うまくいく例と崩れる例の両方をお見せします。"
             "キーワードは多様性と協調ですが、正直、いいことばかりではなかった、というのが今日の裏テーマです。")


# ============================================================
# 章扉A
# ============================================================
def div_background():
    s = section_divider(prs, 1, "背景と問い",
                        "なぜ「小さなAIを話し合わせる」のか。先行研究は何と言っているか。")
    notes(s, "まずは背景です。なぜ小さなAIをわざわざ話し合わせるのか、"
             "そして世の中の研究がこれについて何と言っているかを、かいつまんで説明します。")


# ============================================================
# 2. 大きくすればいい？ → 集団の協調
# ============================================================
def sl_bg_scale():
    s = add_slide(prs)
    title_head(s, "背景①", "モデルは「大きくすれば勝ち」なのか？")
    para_block(s, 0.85, 1.95, 7.0, 3.6, [
        [("これまでのAIは、", 17, INK, FONT, {}),
         ("大きくするほど賢くなってきました", 17, INK, FONT, {"bold": True}),
         ("。", 17, INK, FONT, {})],
        [("でも大きなモデルは、動かすのにお金も電力もかかる。"
          "際限なく大きくする道には限界があります。", 16, INK, FONT, {})],
        [("そこで別の方向として、", 16, INK, FONT, {}),
         ("小さなモデルを何体か集めて協力させる", 16, ACCENT, FONT, {"bold": True}),
         ("という考え方が注目されています。", 16, INK, FONT, {})],
        [("小さいモデルなら、手元の安いパソコンでも複数同時に動かせる。"
          "“数の力”で単体の限界を補えないか、というわけです。", 15.5, SUB, FONT, {})],
    ], space_after=13)
    # 右に対比イラスト（箱）
    rect(s, 8.4, 2.15, 3.9, 1.5, fill=CARD, line=LINE, line_w=1.1,
         shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06, shadow=True)
    text(s, 8.4, 2.28, 3.9, 0.5, [("これまで", 12, SUB, FONT_SB, {})], align=PP_ALIGN.CENTER)
    text(s, 8.4, 2.72, 3.9, 0.8, [("🧠 1体の巨大モデル", 17, INK, FONT, {"bold": True})],
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    text(s, 8.4, 3.9, 3.9, 0.5, [("↓", 22, ACCENT, FONT, {"bold": True})], align=PP_ALIGN.CENTER)
    rect(s, 8.4, 4.35, 3.9, 1.7, fill=TEAL, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
         radius=0.06, shadow=True)
    text(s, 8.4, 4.5, 3.9, 0.5, [("この研究", 12, RGBColor(0xD9,0xE6,0xEC), FONT_SB, {})],
         align=PP_ALIGN.CENTER)
    text(s, 8.4, 4.95, 3.9, 0.9, [("🤝 小さなAI×3体\nで話し合う", 16, CARD, FONT, {"bold": True})],
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    foot(s)
    notes(s, "AIはこれまで、大きくするほど賢くなる、という流れで発展してきました。"
             "でも大きなモデルはお金も電力もかかって、無限には大きくできません。"
             "そこで注目されているのが、小さなモデルを何体か集めて協力させる方向です。"
             "小さければ安いパソコンでも複数動かせるので、数の力で単体の限界を補えないか、という発想ですね。"
             "この研究はまさに、小さなAIを3体話し合わせる、という立場に立っています。")


# ============================================================
# 3. 議論で賢くなる説（Du 2023）
# ============================================================
def sl_bg_debate():
    s = add_slide(prs)
    title_head(s, "背景②", "「AI同士で議論させると賢くなる」という報告")
    para_block(s, 0.85, 1.9, 7.1, 2.5, [
        [("2023年の有名な研究では、", 16, INK, FONT, {}),
         ("複数のAIに同じ問題を解かせ、"
          "互いの答えを見せ合って考え直させる", 16, INK, FONT, {"bold": True}),
         ("と、正答率が上がると報告されました。", 16, INK, FONT, {})],
        [("たとえば、ある試験で 63.9% → 71.1% へ。"
          "「一人で悩むより、視点の違う相手と話した方が良い」——"
          "人間にも通じる直感です。", 15.5, SUB, FONT, {})],
    ], space_after=14)
    # 引用カード
    rect(s, 0.85, 4.7, 11.5, 1.5, fill=CRITIC_BG,
         shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.05)
    rect(s, 0.85, 4.7, 0.13, 1.5, fill=CRITIC, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.5)
    text(s, 1.25, 4.95, 10.8, 1.1, [
        ("先行研究が言っていること（かみくだくと）", 12.5, CRITIC, FONT_SB, {}),
        ("\n「一人の反省だと同じ考えにこだわりがち。"
         "違う視点の相手とぶつけると、“考えの行き詰まり”から抜け出せる」", 16, INK, FONT, {}),
    ], line_spacing=1.2)
    foot(s)
    notes(s, "その協力のやり方で、特に有名なのが“議論”です。"
             "2023年の研究で、複数のAIに同じ問題を解かせて、互いの答えを見せ合って考え直させると、"
             "正答率が上がると報告されました。ある試験で64%から71%へ、といった具合です。"
             "一人で悩むより、視点の違う相手と話した方がいい。人間の感覚にも合いますよね。"
             "別の研究でも、一人だと同じ考えにこだわってしまうけれど、"
             "違う視点をぶつけると行き詰まりから抜け出せる、と言われています。")


# ============================================================
# 4. でも本当に？（3つの批判）
# ============================================================
def sl_bg_critique():
    s = add_slide(prs)
    title_head(s, "背景③", "——ところが「そんなに良くない」という反論も強い")
    items = [
        ("計算量をそろえると勝てない", NEG,
         "議論は何度もAIを呼ぶので手間がかかる。"
         "同じ手間を“ただ何回も解いて多数決”に使うと、そちらの方が強い、という指摘。"),
        ("相手に流されて間違える", GOLD,
         "自信のある正解でも、多数派の空気に合わせて誤答に書き換えてしまう"
         "（“同調”による誤りの伝染）。"),
        ("似た者同士だと意味が薄い", SUB,
         "同じモデルのコピー同士で議論しても新しい視点が出ず、"
         "非効率で不安定になりやすい。"),
    ]
    y = 2.05
    for hd, col, body in items:
        rect(s, 0.85, y, 11.5, 1.32, fill=CARD, line=LINE, line_w=1.0,
             shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.05, shadow=True)
        rect(s, 0.85, y, 0.13, 1.32, fill=col, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.5)
        text(s, 1.2, y + 0.16, 11.0, 0.5, [(hd, 16.5, col, FONT, {"bold": True})])
        text(s, 1.2, y + 0.66, 10.9, 0.6, [(body, 13.5, INK, FONT, {})], line_spacing=1.15)
        y += 1.48
    text(s, 0.85, 6.55, 11.6, 0.5,
         [("→ とくに ", 14, INK, FONT, {}),
          ("小さいモデルでは、素朴な議論はうまくいかない可能性が高い", 14, NEG, FONT, {"bold": True}),
          ("。ここが出発点。", 14, INK, FONT, {})])
    foot(s)
    notes(s, "ところが、そう単純じゃないぞ、という反論も強いんです。3つ挙げます。"
             "1つ目。議論は何度もAIを呼ぶので手間がかかる。その同じ手間を、ただ何回も解いて多数決するのに使うと、"
             "そっちの方が強い、という指摘。これが一番手強い批判です。"
             "2つ目。自信のある正解でも、多数派の空気に流されて誤答に書き換えてしまう。同調による誤りの伝染ですね。"
             "3つ目。同じモデルのコピー同士で議論しても、新しい視点が出ないので意味が薄い。"
             "とくに小さいモデルでは、素朴に議論させてもうまくいかない可能性が高い。"
             "私の研究はまさにここから出発しています。")


# ============================================================
# 5. 本研究の問い
# ============================================================
def sl_bg_gap():
    s = add_slide(prs)
    title_head(s, "背景④・本研究の問い", "既存研究は「エージェントを固定」して議論のやり方だけ工夫してきた")
    para_block(s, 0.85, 1.95, 11.4, 1.5, [
        [("これまでの改善は、", 16, INK, FONT, {}),
         ("ラウンド数や集約方法など“議論の進め方”をいじる", 16, INK, FONT, {"bold": True}),
         ("ものが中心でした。", 16, INK, FONT, {})],
        [("でも、議論の良し悪しは", 16, INK, FONT, {}),
         ("参加するメンバー自身の質と多様性", 16, ACCENT, FONT, {"bold": True}),
         ("に強く左右されるはず。だったら——", 16, INK, FONT, {})],
    ], space_after=12)
    rect(s, 0.85, 4.0, 11.5, 2.1, fill=TEAL, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
         radius=0.05, shadow=True)
    text(s, 1.3, 4.35, 10.7, 1.6, [
        ("本研究の問い", 14, RGBColor(0xCF,0xE1,0xE9), FONT_SB, {}),
        ("\nメンバー（AI）自身を、"
         "“チームへの貢献度”を手がかりに世代交代で鍛えられないか？", 21, CARD, FONT, {"bold": True}),
    ], line_spacing=1.25)
    foot(s)
    notes(s, "ここが問いの核心です。これまでの改善は、ラウンド数を増やすとか集約の仕方を変えるとか、"
             "議論の進め方をいじるものが中心でした。メンバー自身は固定していたんです。"
             "でも議論の良し悪しって、結局は参加するメンバーの質と多様性で決まるはずですよね。"
             "だったら、メンバーであるAI自身を、チームへの貢献度を手がかりに世代交代で鍛えられないか。"
             "これが本研究の問いです。")


# ============================================================
# 章扉B：提案手法
# ============================================================
def div_method():
    s = section_divider(prs, 2, "提案手法",
                        "3つの性格をどう作り、どう議論させ、どう“進化”させるか。")
    notes(s, "では、具体的にどういう仕組みなのかを説明します。"
             "3つの性格をどう作って、どう議論させて、どう進化させるか、の3点です。")


# 手法の全体像（図解を pptx 図形で描く）
def sl_method_overview():
    s = add_slide(prs)
    title_head(s, "手法①", "全体像：1回の議論と、世代をまたぐ進化")
    # ---- 上段：1回の議論の流れ ----
    text(s, 0.85, 1.7, 11.6, 0.4, [("① 1回の議論（推論のとき）", 14, TEAL, FONT_SB, {})])
    # 土台モデル
    rect(s, 0.9, 2.35, 1.95, 1.35, fill=INK_PANEL, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
         radius=0.08, shadow=True)
    text(s, 0.9, 2.5, 1.95, 1.05, [("土台モデル", 13, CARD, FONT_SB, {}),
         ("\nQwen3-4B\n（共通の1体）", 11, RGBColor(0xC7,0xCE,0xD4), FONT, {})],
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, line_spacing=1.15)
    # 3つの性格
    roles = [("批判", CRITIC, CRITIC_BG), ("実務", PRAG, PRAG_BG), ("探索", EXPLORE, EXPLORE_BG)]
    for i, (nm, col, bg) in enumerate(roles):
        yy = 2.18 + i * 0.56
        rect(s, 3.55, yy, 1.5, 0.46, fill=bg, line=col, line_w=1.2,
             shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.4)
        text(s, 3.55, yy, 1.5, 0.46, [(f"性格：{nm}", 12, col, FONT_SB, {})],
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    line(s, 2.9, 3.02, 0.6, 0, color=SUB, weight=1.3)
    # 議論
    rect(s, 5.5, 2.35, 1.85, 1.35, fill=CARD, line=TEAL, line_w=1.4,
         shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08, shadow=True)
    text(s, 5.5, 2.5, 1.85, 1.05, [("議論", 14, TEAL, FONT_SB, {}),
         ("\n答えを見せ合い\n2ラウンド", 11, INK, FONT, {})],
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, line_spacing=1.15)
    text(s, 5.05, 2.85, 0.5, 0.5, [("→", 20, SUB, FONT, {"bold": True})], align=PP_ALIGN.CENTER)
    # 集約
    rect(s, 8.0, 2.35, 1.85, 1.35, fill=CARD, line=LINE, line_w=1.2,
         shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08, shadow=True)
    text(s, 8.0, 2.5, 1.85, 1.05, [("答えを集約", 13, INK, FONT_SB, {}),
         ("\n重み付き投票\nで最終回答", 11, SUB, FONT, {})],
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, line_spacing=1.15)
    text(s, 7.45, 2.85, 0.5, 0.5, [("→", 20, SUB, FONT, {"bold": True})], align=PP_ALIGN.CENTER)
    text(s, 10.05, 2.85, 2.2, 0.5, [("→ 最終回答", 15, INK, FONT, {"bold": True})],
         anchor=MSO_ANCHOR.MIDDLE)
    # ---- 下段：進化ループ ----
    line(s, 0.85, 4.35, 11.6, 0, color=LINE, weight=1.0)
    text(s, 0.85, 4.5, 11.6, 0.4, [("② 世代をまたぐ進化（学習の外側で繰り返す）", 14, ACCENT, FONT_SB, {})])
    steps = [("たくさんの候補で\n議論を評価", INK_PANEL, CARD),
             ("チームへの貢献度\nを測る（Shapley値）", TEAL, CARD),
             ("貢献の高い個体を\n掛け合わせ＋変異", ACCENT, CARD),
             ("次の世代へ", POS, CARD)]
    x = 0.9
    for i, (tx, col, tc) in enumerate(steps):
        rect(s, x, 5.05, 2.5, 1.1, fill=col, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
             radius=0.08, shadow=True)
        text(s, x, 5.05, 2.5, 1.1, [(tx, 12.5, tc, FONT_SB, {})],
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, line_spacing=1.15)
        if i < 3:
            text(s, x + 2.5, 5.3, 0.42, 0.6, [("→", 18, SUB, FONT, {"bold": True})],
                 align=PP_ALIGN.CENTER)
        x += 2.92
    text(s, 0.9, 6.35, 11.5, 0.5, [("↺　この②を何世代も回して、"
         "「一緒に組んだときに強いメンバー」を育てる", 13, SUB, FONT, {})])
    foot(s)
    notes(s, "全体像です。上下2段に分けました。"
             "上段が“1回の議論”。共通の土台モデル1体に、批判・実務・探索という3つの性格を持たせます。"
             "この3体が同じ問題を解いて、答えを見せ合いながら2ラウンド議論し、最後に重み付きの投票で結論を出す。"
             "下段が“進化”。これは学習の外側で繰り返すループです。"
             "たくさんの候補で議論を評価し、それぞれがチームにどれだけ貢献したかを測り、"
             "貢献の高い個体を掛け合わせて少し変異させ、次の世代を作る。"
             "これを何世代も回して、一緒に組んだときに強いメンバーを育てる、というのが狙いです。")


def sl_personas():
    s = add_slide(prs)
    title_head(s, "手法②", "3つの性格 —— わざと“違う考え方”をさせる")
    data = [
        ("批判（けんしょう）", CRITIC, CRITIC_BG,
         "反証・例外・境界を疑う", "「本当に？ 反例はないか」と粗を探す役"),
        ("実務（じつむ）", PRAG, PRAG_BG,
         "実現性とコストで判断", "「結局どれが使えるか」と地に足をつける役"),
        ("探索（たんさく）", EXPLORE, EXPLORE_BG,
         "仮説と比喩で発想を広げる", "「別の見方はないか」と選択肢を増やす役"),
    ]
    x0, w, gap = 0.85, 3.72, 0.19
    for i, (nm, col, bg, tag, role) in enumerate(data):
        x = x0 + i * (w + gap)
        rect(s, x, 2.1, w, 3.5, fill=CARD, line=LINE, line_w=1.0,
             shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.05, shadow=True)
        rect(s, x, 2.1, w, 0.9, fill=col, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.05)
        rect(s, x, 2.6, w, 0.4, fill=col)
        text(s, x, 2.24, w, 0.7, [(nm, 16.5, CARD, FONT_SB, {})],
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        text(s, x + 0.3, 3.2, w - 0.6, 0.5, [(tag, 13.5, col, FONT, {"bold": True})],
             align=PP_ALIGN.CENTER)
        line(s, x + 0.5, 3.72, w - 1.0, 0, color=LINE, weight=1.0)
        text(s, x + 0.32, 3.95, w - 0.62, 1.5, [(role, 13, INK, FONT, {})],
             align=PP_ALIGN.CENTER, line_spacing=1.25)
    text(s, 0.85, 5.95, 11.6, 0.9,
         [("実際には、これらは短い日本語の指示文で与えます（付録に原文）。"
           "大事なのは ", 13.5, INK, FONT, {}),
          ("“同じ問題を、あえて違う切り口で見る3体”", 13.5, ACCENT, FONT, {"bold": True}),
          ("を用意したこと。", 13.5, INK, FONT, {})])
    foot(s)
    notes(s, "3つの性格を紹介します。批判は、反証や例外を疑って粗を探す役。"
             "実務は、実現性やコストで、結局どれが使えるかを判断する役。"
             "探索は、仮説や比喩で別の見方を出して選択肢を増やす役です。"
             "実際にはこれらを短い日本語の指示文で与えているだけなんですが、"
             "大事なのは、同じ問題をあえて違う切り口で見る3体を用意した、という点です。"
             "この直後のデモで、この3つの個性が実際に効いている様子が見られます。")


def sl_shapley():
    s = add_slide(prs)
    title_head(s, "手法③", "貢献度の測り方：「その人が抜けたら、どれだけ困る？」")
    para_block(s, 0.85, 1.9, 6.6, 3.4, [
        [("進化で“良い個体”を選ぶには、"
          "個人の成績ではなく", 16, INK, FONT, {}),
         ("チームへの貢献", 16, ACCENT, FONT, {"bold": True}),
         ("を測りたい。", 16, INK, FONT, {})],
        [("そこで、", 15.5, INK, FONT, {}),
         ("あらゆる組み合わせでチームを試し、"
          "「その1体が居るときと居ないときの差」を平均", 15.5, INK, FONT, {"bold": True}),
         ("します。", 15.5, INK, FONT, {})],
        [("これは協力ゲーム理論の", 15, INK, FONT, {}),
         ("シャープレイ値", 15, TEAL, FONT, {"bold": True}),
         ("という考え方。3体なら全7通りを実測でき、"
          "近似なしで公平に貢献度を配れます。", 15, INK, FONT, {})],
    ], space_after=13)
    # 右：実例カード
    rect(s, 7.75, 1.95, 4.6, 3.55, fill=PRAG_BG, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
         radius=0.05)
    text(s, 8.05, 2.15, 4.0, 0.5, [("実際に起きたこと", 13, PRAG, FONT_SB, {})])
    para_block(s, 8.05, 2.65, 4.05, 2.8, [
        [("ある個体は、", 14, INK, FONT, {}),
         ("単体の成績は3体で最低", 14, NEG, FONT, {"bold": True}),
         ("。", 14, INK, FONT, {})],
        [("でも", 14, INK, FONT, {}),
         ("チームに入れると成績が最も伸びた", 14, POS, FONT, {"bold": True}),
         ("。", 14, INK, FONT, {})],
        [("→ 個人の点数だけ見ていたら"
          "捨てていた個体を、", 13.5, INK, FONT, {}),
         ("貢献度は正しく拾い上げた", 13.5, TEAL, FONT, {"bold": True}),
         ("。", 13.5, INK, FONT, {})],
    ], space_after=12, line_spacing=1.25)
    text(s, 0.85, 6.15, 11.6, 0.7, [("“優秀な個人の寄せ集め＝最強のチーム”とは限らない"
         "——アンサンブル学習の古典的な教訓とも一致します。", 13.5, SUB, FONT, {})])
    foot(s)
    notes(s, "進化で良い個体を選ぶとき、個人の成績ではなく、チームへの貢献を測りたいんです。"
             "そこで、あらゆる組み合わせでチームを試して、その1体が居るときと居ないときの差を平均します。"
             "これは協力ゲーム理論のシャープレイ値という考え方で、3体なら全7通りを実際に測れるので、"
             "近似なしで公平に貢献度を配れます。"
             "実際、面白いことが起きました。単体の成績は3体で最低なのに、"
             "チームに入れると一番成績が伸びる個体があったんです。"
             "個人の点数だけ見ていたら捨てていた個体を、貢献度はちゃんと拾い上げた。"
             "優秀な個人の寄せ集めが最強のチームとは限らない、という教訓そのものでした。")


def sl_evolution():
    s = add_slide(prs)
    title_head(s, "手法④", "進化のしくみ：重みを“遺伝子”とみなす")
    para_block(s, 0.85, 1.95, 11.4, 1.4, [
        [("各AIの“性格”は、土台モデルに足す小さな重み（LoRA）で表せます。"
          "この重みを遺伝子とみなし、", 16, INK, FONT, {}),
         ("良い2体を掛け合わせ（交叉）、少しゆらす（突然変異）", 16, ACCENT, FONT, {"bold": True}),
         ("。", 16, INK, FONT, {})],
    ], space_after=10)
    ops = [
        ("交叉", TEAL, "貢献度の高い2体の重みを混ぜ、"
         "両方の“良さ”を引き継いだ子を作る"),
        ("突然変異", ACCENT, "重みをランダムに少しだけ変え、"
         "新しい可能性を試す"),
        ("選択", POS, "チーム貢献度（＋多様性）の高い個体を"
         "次の世代に残す"),
    ]
    x0, w, gap = 0.85, 3.72, 0.19
    for i, (nm, col, body) in enumerate(ops):
        x = x0 + i * (w + gap)
        rect(s, x, 3.5, w, 2.05, fill=CARD, line=LINE, line_w=1.0,
             shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06, shadow=True)
        rect(s, x + 0.32, 3.75, 1.4, 0.5, fill=col, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.4)
        text(s, x + 0.32, 3.75, 1.4, 0.5, [(nm, 14, CARD, FONT_SB, {})],
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        text(s, x + 0.34, 4.4, w - 0.66, 1.0, [(body, 13, INK, FONT, {})], line_spacing=1.25)
        x += 0
    text(s, 0.85, 6.05, 11.6, 0.8,
         [("多様性は成績に足し算せず、", 13.5, INK, FONT, {}),
          ("“似た者同士は割り引く”", 13.5, ACCENT, FONT, {"bold": True}),
          ("という形で保ちます（理論的に妥当な入れ方）。", 13.5, INK, FONT, {})])
    foot(s)
    notes(s, "進化の中身です。各AIの性格は、土台モデルに足す小さな重み、いわゆるLoRAで表せます。"
             "この重みを遺伝子とみなして、良い2体を掛け合わせ、少しゆらす。生き物の進化と同じ発想です。"
             "交叉は、貢献度の高い2体の重みを混ぜて、両方の良さを引き継いだ子を作る。"
             "突然変異は、重みをランダムに少し変えて新しい可能性を試す。"
             "選択は、チーム貢献度の高い個体を次の世代に残す。"
             "多様性については、成績に足し算するのではなく、似た者同士は割り引く、という形で保ちます。"
             "ここは理論的な裏づけがある入れ方をしています。")


# ============================================================
# 章扉C：デモ
# ============================================================
def div_demo():
    s = section_divider(prs, 3, "実際の会話を見る",
                        "ここからは本物のログ。AIたちが議論して、正解にたどり着く（時に、崩れる）。")
    notes(s, "ここからが今日のメインです。実際にAIたちがどう会話しているか、本物のログをお見せします。"
             "まずうまくいく例、次に崩れてしまう例、両方見ていきます。スライドを送るごとに会話が進みます。")


def sl_demo_intro():
    s = add_slide(prs)
    title_head(s, "デモの見方", "3体が同じ問題を解き、答えを見せ合って考え直す")
    steps = [
        ("① 各自が解く", "3体が別々に問題を解き、それぞれ答えを出す", INK_PANEL),
        ("② 見せ合う", "互いの答えと理由を読み、もう一度考え直す", TEAL),
        ("③ まとめる", "最終的な答えを多数決（重み付き投票）で決める", POS),
    ]
    x0, w, gap = 0.85, 3.72, 0.19
    for i, (hd, body, col) in enumerate(steps):
        x = x0 + i * (w + gap)
        rect(s, x, 2.3, w, 2.0, fill=CARD, line=LINE, line_w=1.0,
             shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06, shadow=True)
        text(s, x + 0.32, 2.55, w - 0.6, 0.6, [(hd, 17, col, FONT, {"bold": True})])
        text(s, x + 0.34, 3.25, w - 0.66, 1.0, [(body, 13.5, INK, FONT, {})], line_spacing=1.25)
        if i < 2:
            text(s, x + w - 0.02, 2.95, 0.28, 0.6, [("→", 18, SUB, FONT, {"bold": True})],
                 align=PP_ALIGN.CENTER)
    rect(s, 0.85, 4.7, 11.5, 1.5, fill=PAPER, line=ACCENT, line_w=1.3,
         shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.05)
    text(s, 1.2, 4.95, 10.9, 1.1, [
        ("これから見せるのは、", 14.5, INK, FONT, {}),
        ("すべて実験で実際に記録された会話", 14.5, ACCENT, FONT, {"bold": True}),
        ("です（英語のやりとりを日本語に要約、原文の一部も併記）。\n"
         "最初は“うまくいく例”から。数学の問題です。", 14.5, INK, FONT, {})],
        line_spacing=1.3)
    foot(s)
    notes(s, "デモの見方です。3ステップ。まず3体が別々に問題を解いて答えを出す。"
             "次に互いの答えと理由を読んで、もう一度考え直す。最後に多数決で結論を決める。"
             "これから見せるのは、すべて実験で実際に記録された会話です。"
             "英語のやりとりなので日本語に要約していますが、原文の一部もそのまま載せています。"
             "まずはうまくいく例から。数学の問題です。")


# ---- 成功例（カントール集合）ビルドアップ ----
def _cantor_question(s):
    rect(s, 0.85, 1.72, 11.5, 1.72, fill=CRITIC_BG, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
         radius=0.05)
    text(s, 1.2, 1.9, 11.0, 0.5, [("問題（数学）", 12.5, CRITIC, FONT_SB, {})])
    text(s, 1.2, 2.28, 11.0, 1.1, [
        ("「1/4 と 1/13 は “カントール集合” に入る？」", 18, INK, FONT, {"bold": True}),
        ("\nカントール集合 ＝ ざっくり言うと「3進数で書いたときに “1” が出てこない数」の集まり。", 14, SUB, FONT, {}),
    ], line_spacing=1.25)
    rect(s, 9.9, 1.85, 2.3, 0.62, fill=POS, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.3)
    text(s, 9.9, 1.85, 2.3, 0.62, [("正解：両方入る", 13.5, CARD, FONT_SB, {})],
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)


def sl_cantor_q():
    s = add_slide(prs)
    title_head(s, "デモ・成功例 ①／④", "まず問題を確認する")
    _cantor_question(s)
    # 3体が考え中
    for i, (nm, col, bg) in enumerate([("批判", CRITIC, CRITIC_BG),
                                       ("実務", PRAG, PRAG_BG), ("探索", EXPLORE, EXPLORE_BG)]):
        x = 0.85 + i * 3.9
        rect(s, x, 4.0, 3.7, 2.15, fill=CARD, line=LINE, line_w=1.0,
             shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06, shadow=True)
        rect(s, x + 0.3, 4.25, 1.5, 0.44, fill=bg, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.4)
        text(s, x + 0.3, 4.25, 1.5, 0.44, [(nm, 12.5, col, FONT_SB, {})],
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        text(s, x, 5.0, 3.7, 0.9, [("考え中…", 15, FAINT, FONT, {})],
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    foot(s)
    notes(s, "問題はこれです。4分の1と13分の1が、カントール集合に入るか。"
             "カントール集合というのは、ざっくり言うと、3進数で書いたときに1が出てこない数の集まりです。"
             "正解は、実は両方とも入る、なんですが、これがなかなか難しい。3体がどう答えるか見てみましょう。")


def _cantor_r1_bubbles(s):
    b = [
        ("批判", CRITIC, CRITIC_BG, "1/13 は3進数で1が出る…と判断", [
            [("1/4 は入る。でも 1/13 は3進数にすると "
              "“1” が出るはず → 入らない", 13, INK, FONT, {})],
            [("“…contains a 1, so 1/13 does not belong.”", 10.5, FAINT, FONT_L, {"italic": True})],
        ], "J", False),
        ("実務", PRAG, PRAG_BG, "展開して級数で検算 → 入る", [
            [("1/4 は入る。1/13 も 0.00202…(3進) と展開でき、"
              "足し合わせると確かに 1/13。だから入る", 13, INK, FONT, {})],
            [("“= 2/26 = 1/13 … only 0s and 2s → belongs.”", 10.5, FAINT, FONT_L, {"italic": True})],
        ], "H", True),
        ("探索", EXPLORE, EXPLORE_BG, "批判と同じく“1が出る”と判断", [
            [("1/13 の3進展開に “1” が見える → 入らない、と判断", 13, INK, FONT, {})],
            [("“…contains a 1 … answers are [1, 0].”", 10.5, FAINT, FONT_L, {"italic": True})],
        ], "J", False),
    ]
    y = 1.78
    for nm, col, bg, head, body, ans, ok in b:
        bubble(s, 0.85, y, 11.5, 1.42, nm, col, bg, head, body, answer=ans, answer_ok=ok)
        y += 1.56


def sl_cantor_r1():
    s = add_slide(prs)
    title_head(s, "デモ・成功例 ②／④", "① 各自が解く —— この時点では“不正解”が多数派")
    _cantor_r1_bubbles(s)
    rect(s, 0.85, 6.5, 11.5, 0.5, fill=NEG, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.3)
    text(s, 0.85, 6.5, 11.5, 0.5, [
        ("いま多数決すると… 不正解の「J」が2票。正解は“実務”1体だけに埋もれている", 13, CARD, FONT_SB, {})],
        align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    foot(s)
    notes(s, "各自が解きました。批判と探索は、1/13を3進数にすると1が出るから入らない、と判断してJ。"
             "ところが実務だけは違った。1/13を0.00202…と展開して、しかも足し合わせると本当に1/13になると検算までして、"
             "だから入る、と答えてH。これが正解なんです。"
             "でもこの時点で多数決すると、不正解のJが2票。正解が1体だけに埋もれている。あぶない状況です。"
             "普通の多数決だとここで負けます。ここから議論が始まります。")


def sl_cantor_r2():
    s = add_slide(prs)
    title_head(s, "デモ・成功例 ③／④", "② 見せ合う —— 検算を見た2体が“訂正”した")
    b = [
        ("批判", CRITIC, CRITIC_BG, "実務の検算を自分でも確かめた → 訂正", [
            [("2/27 + 2/729 + … を自分で計算したら確かに 1/13。"
              "“1” は出ない。答えを H に変える", 13, INK, FONT, {})],
        ], "H", True),
        ("探索", EXPLORE, EXPLORE_BG, "批判の展開ミスに気づいた → 訂正", [
            [("さっきの3進展開は誤り。正しくは 0.00202… で “1” を含まない。"
              "H に変える", 13, INK, FONT, {})],
        ], "H", True),
        ("実務", PRAG, PRAG_BG, "そのまま（正解を主張し続ける）", [
            [("展開と検算は合っている。両方カントール集合に入る＝H", 13, INK, FONT, {})],
        ], "H", True),
    ]
    y = 2.0
    for nm, col, bg, head, body, ans, ok in b:
        bubble(s, 0.85, y, 11.5, 1.18, nm, col, bg, head, body, answer=ans, answer_ok=ok)
        y += 1.34
    rect(s, 0.85, 6.5, 11.5, 0.5, fill=POS, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.3)
    text(s, 0.85, 6.5, 11.5, 0.5, [
        ("正解の「H」が多数派に逆転 —— 少数派だった正解が、検算を通じてチームを動かした", 13, CARD, FONT_SB, {})],
        align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    foot(s)
    notes(s, "議論のラウンドです。批判は、実務の検算を自分でも確かめて、確かに1は出ない、と納得してHに訂正しました。"
             "探索も、さっきの3進展開が間違っていた、正しくは1を含まない、と気づいてHに訂正。"
             "実務はそのまま正解のHを主張。結果、正解のHが多数派に逆転したんです。"
             "少数派だった正解が、検算という“確かめられる根拠”を通じてチーム全体を動かした。これが議論のいい面です。"
             "――ちなみに実際のログでは実務が一瞬迷う場面もあるんですが、大筋はこの通り、正解に収束しました。")


def sl_cantor_msg():
    s = add_slide(prs)
    title_head(s, "デモ・成功例 ④／④", "この例が示すこと")
    para_block(s, 0.85, 2.1, 11.4, 2.0, [
        [("多数決だけなら負けていた問題を、", 18, INK, FONT, {}),
         ("議論がひっくり返して正解にした", 18, POS, FONT, {"bold": True}),
         ("。", 18, INK, FONT, {})],
    ], space_after=10)
    rect(s, 0.85, 3.3, 11.5, 2.4, fill=PRAG_BG, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.05)
    text(s, 1.25, 3.6, 10.8, 2.0, [
        ("なぜうまくいったのか", 13.5, PRAG, FONT_SB, {}),
        ("\n数学は「答えを確かめられる」領域だから。", 17, INK, FONT, {"bold": True}),
        ("\n相手の主張が正しいかを自分で検算できるので、"
         "議論が“なんとなくの同調”ではなく“根拠にもとづく訂正”になる。", 15, INK, FONT, {}),
        ("\n逆に言えば、確かめにくい問題では、この良さは出にくい——次はその例。", 14.5, SUB, FONT, {}),
    ], line_spacing=1.3)
    foot(s)
    notes(s, "この例が示すのは、多数決だけなら負けていた問題を、議論がひっくり返して正解にした、ということです。"
             "なぜうまくいったか。数学は答えを確かめられる領域だからです。"
             "相手の主張が正しいかを自分で検算できるので、議論がなんとなくの同調ではなく、根拠にもとづく訂正になる。"
             "逆に言えば、確かめにくい問題では、この良さは出にくい。次はその、崩れてしまう例を見ます。")


# ---- 失敗例（暗号）ビルドアップ ----
def sl_crypto_q():
    s = add_slide(prs)
    title_head(s, "デモ・失敗例 ①／③", "次は“崩れる”例 —— 常識的な問題")
    rect(s, 0.85, 1.72, 11.5, 1.5, fill=EXPLORE_BG, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.05)
    text(s, 1.2, 1.92, 11.0, 0.5, [("問題（情報・常識）", 12.5, EXPLORE, FONT_SB, {})])
    text(s, 1.2, 2.32, 11.0, 0.9, [
        ("「送信側で暗号化されたメッセージは、どこで復号（元に戻す）される？」", 18, INK, FONT, {"bold": True}),
    ], line_spacing=1.2)
    rect(s, 9.9, 1.85, 2.3, 0.62, fill=POS, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.3)
    text(s, 9.9, 1.85, 2.3, 0.62, [("正解：受信側", 13.5, CARD, FONT_SB, {})],
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    # R1: 正解2体
    b = [
        ("批判", CRITIC, CRITIC_BG, "受信者が中身を読む必要 → 受信側", [
            [("暗号化は送信側。中身を読むのは受け取る側だから、復号は受信側", 13, INK, FONT, {})]],
         "受信側", True),
        ("探索", EXPLORE, EXPLORE_BG, "宛先＝受信側（F か H で少し迷う）", [
            [("宛先で使える必要がある → 受信側。ただ“端末側”とも言えるか…", 13, INK, FONT, {})]],
         "受信側", True),
        ("実務", PRAG, PRAG_BG, "端末＝クライアント側では？", [
            [("実際に復号するのは受け手の端末。クライアント側だと思う", 13, INK, FONT, {})]],
         "端末側", False),
    ]
    y = 3.32
    for nm, col, bg, head, body, ans, ok in b:
        bubble(s, 0.85, y, 11.5, 0.9, nm, col, bg, head, body, answer=ans, answer_ok=ok)
        y += 1.0
    rect(s, 0.85, 6.45, 11.5, 0.5, fill=POS, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.3)
    text(s, 0.85, 6.45, 11.5, 0.5, [("① 各自が解く：正解「受信側」が2票 —— ここまでは正しい", 13, CARD, FONT_SB, {})],
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    foot(s)
    notes(s, "次は崩れる例です。問題はシンプル。送信側で暗号化されたメッセージは、どこで復号されるか。"
             "正解は受信側です。各自が解くと、批判と探索は受信側と答えて正解。"
             "実務だけが、実際に復号するのは受け手の端末だからクライアント側だ、と答えて外した。"
             "この時点では正解の受信側が2票。ここまでは正しいんです。問題は次のラウンドで起きます。")


def sl_crypto_r2():
    s = add_slide(prs)
    title_head(s, "デモ・失敗例 ②／③", "② 見せ合う —— “もっともらしい深読み”に流された")
    b = [
        ("批判", CRITIC, CRITIC_BG, "よく考えると端末側かも… → 転向", [
            [("実際に復号するのは端末＝クライアント側では、と考え直す。答えを変える", 13, INK, FONT, {})]],
         "端末側", False),
        ("探索", EXPLORE, EXPLORE_BG, "アプリの例を思いつき → 転向", [
            [("SignalやWhatsAppはスマホ側で復号する。だから“端末側”が精密だ", 13, INK, FONT, {})],
            [("“…decrypt on the user’s phone … client site is more precise.”",
              10.5, FAINT, FONT_L, {"italic": True})]],
         "端末側", False),
        ("実務", PRAG, PRAG_BG, "受信側でよい（正解を維持）", [
            [("宛先＝受信側で問題ない", 13, INK, FONT, {})]],
         "受信側", True),
    ]
    y = 3.2
    for nm, col, bg, head, body, ans, ok in b:
        h = 1.05 if nm == "探索" else 0.85
        bubble(s, 0.85, y, 11.5, h, nm, col, bg, head, body, answer=ans, answer_ok=ok)
        y += h + 0.12
    rect(s, 0.85, 6.5, 11.5, 0.5, fill=NEG, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.3)
    text(s, 0.85, 6.5, 11.5, 0.5, [
        ("不正解の「端末側」が2票に —— 正しかった多数派が、深読みで崩れた", 13, CARD, FONT_SB, {})],
        align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    foot(s)
    notes(s, "議論のラウンドです。ここで悪い方向に転がります。"
             "批判が、よく考えると実際に復号するのは端末側では、と考え直して転向。"
             "探索も、SignalやWhatsAppはスマホ側で復号する、だから端末側が精密だ、という例を思いついて転向。"
             "実務だけが受信側を維持しました。結果、不正解の端末側が2票になってしまった。"
             "“もっともらしい深読み”に引きずられて、正しかった多数派が崩れたんです。"
             "確かめられない問題だと、賢そうな理屈ほど危ない、という典型例です。")


def sl_crypto_msg():
    s = add_slide(prs)
    title_head(s, "デモ・失敗例 ③／③", "2つのデモが示すこと")
    cols = [
        ("成功（数学）", POS, PRAG_BG,
         ["答えを確かめられる", "検算で“根拠ある訂正”", "少数派の正解が勝った"]),
        ("失敗（常識）", NEG, RGBColor(0xF6,0xEA,0xE6),
         ["答えを確かめにくい", "深読みで“同調”が起きる", "正しい多数派が崩れた"]),
    ]
    for i, (hd, col, bg, items) in enumerate(cols):
        x = 0.85 + i * 5.95
        rect(s, x, 2.15, 5.55, 2.9, fill=bg, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.05)
        rect(s, x, 2.15, 5.55, 0.66, fill=col, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.05)
        rect(s, x, 2.48, 5.55, 0.33, fill=col)
        text(s, x, 2.15, 5.55, 0.66, [(hd, 16, CARD, FONT_SB, {})],
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        bullets(s, x + 0.4, 3.05, 4.9, 1.9, items, size=14.5, gap=11,
                marker="●", marker_color=col)
    rect(s, 0.85, 5.35, 11.5, 1.35, fill=INK_PANEL, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.05)
    text(s, 1.25, 5.6, 10.8, 0.95, [
        ("同じ“議論”でも、結果は問題の性質で正反対になる。", 17, CARD, FONT, {"bold": True}),
        ("\nこの「領域による効き方の違い」こそ、実験でも一番はっきり出た発見でした。",
         14, RGBColor(0xC7,0xCE,0xD4), FONT, {})],
        line_spacing=1.3)
    foot(s)
    notes(s, "2つのデモをまとめます。成功した数学の例は、答えを確かめられるので、検算による根拠ある訂正が起きて、"
             "少数派の正解が勝ちました。失敗した常識の例は、答えを確かめにくいので、深読みによる同調が起きて、"
             "正しい多数派が崩れた。同じ議論でも、結果は問題の性質で正反対になるんです。"
             "この、領域による効き方の違いこそ、実験でも一番はっきり出た発見でした。ここから結果の話に移ります。")


# ============================================================
# 章扉D：実験と結果
# ============================================================
def div_results():
    s = section_divider(prs, 4, "実験と結果",
                        "何を、どんな環境で測ったか。そして、正直な結果。")
    notes(s, "ここからは実験と結果です。何を、どんな環境で測ったか。そして正直な結果を報告します。")


def sl_setup():
    s = add_slide(prs)
    title_head(s, "実験設定", "何を・どこで・どう測ったか")
    boxes = [
        ("使ったAI", TEAL, [
            "Qwen3-4B（40億パラメータ級の小型・公開モデル）",
            "土台は1体、性格ごとに小さな追加重み（LoRA）",
        ]),
        ("計算環境", INK_PANEL, [
            "クラウド上のGPUで学習・評価を実施",
            "途中で止まっても再開できる設計",
        ]),
        ("3種類のテスト", ACCENT, [
            "一般知識・推論（MMLU-Pro）",
            "数学（MATH-500）",
            "大学院レベルの科学（SuperGPQA）",
        ]),
        ("比べた相手", POS, [
            "ベース単体／多数決9回（SC@9）",
            "素の議論／LoRAチーム／進化後チーム",
        ]),
    ]
    x0, y0, w, hh, gx, gy = 0.85, 2.0, 5.62, 1.75, 0.28, 0.25
    for i, (hd, col, items) in enumerate(boxes):
        x = x0 + (i % 2) * (w + gx)
        y = y0 + (i // 2) * (hh + gy)
        rect(s, x, y, w, hh, fill=CARD, line=LINE, line_w=1.0,
             shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06, shadow=True)
        rect(s, x, y, 0.13, hh, fill=col, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.5)
        text(s, x + 0.32, y + 0.16, w - 0.5, 0.4, [(hd, 14.5, col, FONT, {"bold": True})])
        bullets(s, x + 0.34, y + 0.66, w - 0.62, hh - 0.7, items, size=12.5, gap=6,
                marker="・", marker_color=col, line_spacing=1.12)
    text(s, 0.85, 5.95, 11.6, 0.9,
         [("公平さのため、", 13.5, INK, FONT, {}),
          ("同じ問題で条件どうしを“対戦”させ、統計検定（多重比較の補正つき）", 13.5, INK, FONT, {"bold": True}),
          ("で判定。乱数の種を変えても結果が変わらないかも確認しました。", 13.5, INK, FONT, {})],
         line_spacing=1.25)
    foot(s)
    notes(s, "実験設定です。使ったAIは、Qwen3-4Bという40億パラメータ級の小型で公開されているモデル。"
             "土台は1体で、性格ごとに小さな追加重みを載せています。"
             "計算はクラウド上のGPUで行い、途中で止まっても再開できる設計にしました。"
             "テストは3種類。一般知識・推論、数学、大学院レベルの科学です。"
             "比べた相手は、ベース単体、多数決9回、素の議論、LoRAチーム、進化後チーム。"
             "公平さのため、同じ問題で条件どうしを対戦させて、多重比較の補正つきで統計検定しています。"
             "乱数の種を変えても結果が変わらないかも確認しました。ここは厳しめにやっています。")


def sl_finding_domain():
    s = add_slide(prs)
    title_head(s, "発見①", "議論が効く問題・効かない問題がある（領域依存）")
    pic_fit(s, str(AST / "fig_domain_flip.png"), 0.85, 1.75, 6.7, 4.6, align="center")
    para_block(s, 7.8, 2.1, 4.6, 4.0, [
        [("同じ「素の議論」でも、", 15, INK, FONT, {}),
         ("問題の種類で結果が逆転", 15, ACCENT, FONT, {"bold": True}),
         ("しました。", 15, INK, FONT, {})],
        [("数学や科学では議論すると上がる。"
          "でも一般知識では、むしろ下がってしまう。", 14.5, INK, FONT, {})],
        [("鍵は", 14.5, INK, FONT, {}),
         ("「答えを確かめられるか」", 14.5, TEAL, FONT, {"bold": True}),
         ("。確かめられる問題では、議論が正しい方向に働く"
          "（さっきの数学デモと同じ）。", 14.5, INK, FONT, {})],
        [("この“反転”は、先行研究の食い違い"
          "（議論は効く／効かない）も説明できます。", 13.5, SUB, FONT, {})],
    ], space_after=13, line_spacing=1.25)
    foot(s)
    notes(s, "1つ目の発見です。同じ素の議論でも、問題の種類で結果が逆転しました。"
             "数学や科学では議論すると上がる。でも一般知識では、むしろ下がってしまう。"
             "鍵は、答えを確かめられるかどうかです。確かめられる問題では、議論が正しい方向に働く。"
             "さっきの数学デモと同じ理屈ですね。"
             "この反転は、議論は効くという研究と効かないという研究の食い違いも、うまく説明できます。")


def sl_finding_repair():
    s = add_slide(prs)
    title_head(s, "発見②", "“性格”を教えると賢さが削れる —— そして治せる")
    pic_fit(s, str(AST / "fig_cot_repair.png"), 0.85, 1.8, 7.4, 4.3, align="center")
    para_block(s, 8.5, 2.1, 3.9, 4.0, [
        [("性格をむりに教え込むと、", 14.5, INK, FONT, {}),
         ("考える途中の説明が半分に縮み", 14.5, NEG, FONT, {"bold": True}),
         ("、その分だけ数学が解けなくなっていました。", 14.5, INK, FONT, {})],
        [("原因は「短い答え方」を覚えすぎたこと。", 14, INK, FONT, {})],
        [("対策は", 14.5, INK, FONT, {}),
         ("元のモデル自身の“長い解答”を混ぜて学び直す", 14.5, TEAL, FONT, {"bold": True}),
         ("こと。これで賢さを取り戻せました。", 14.5, INK, FONT, {})],
    ], space_after=14, line_spacing=1.28)
    foot(s)
    notes(s, "2つ目の発見です。性格をむりに教え込むと、副作用がありました。"
             "考える途中の説明が半分くらいに縮んでしまって、その分だけ数学が解けなくなっていたんです。"
             "左のグラフのように、思考が2400字から1200字に縮み、右のように数学の正答率が13ポイント落ちた。"
             "原因は、短い答え方を覚えすぎたこと。"
             "対策は、元のモデル自身の長い解答を混ぜて学び直させること。これで賢さを取り戻せました。"
             "壊れた原因が分かったので、それを取り除いた、というわけです。")


def sl_final_1():
    s = add_slide(prs)
    title_head(s, "最終結果 ①", "処方の効果：壊れたチームは“元どおり”まで回復した")
    pic_fit(s, str(AST / "fig_final_results.png"), 0.85, 1.8, 8.3, 4.6, align="center", valign="top")
    para_block(s, 9.35, 2.15, 3.1, 4.2, [
        [("旧チーム（オレンジ）は"
          "ベースより下がっていた。", 13.5, INK, FONT, {})],
        [("処方後の新チーム（青）は"
          "3種目すべてで持ち直し、", 13.5, INK, FONT, {}),
         ("ベース単体と互角", 13.5, POS, FONT, {"bold": True}),
         ("に。", 13.5, INK, FONT, {})],
        [("数学では", 13.5, INK, FONT, {}),
         ("多数決9回（SC@9）に肩を並べた", 13.5, TEAL, FONT, {"bold": True}),
         ("。", 13.5, INK, FONT, {})],
    ], space_after=14, line_spacing=1.3)
    foot(s)
    notes(s, "最終結果です。まず処方の効果から。"
             "オレンジの旧チームは、実はベースより下がっていました。壊れていたんですね。"
             "処方後の青い新チームは、3種目すべてで持ち直して、ベース単体と互角まで回復しました。"
             "特に数学では、最強の相手である多数決9回に肩を並べています。"
             "壊れたチームを元どおりまで治せた、というのがここでの成果です。")


def sl_final_2():
    s = add_slide(prs)
    title_head(s, "最終結果 ②", "正直な結論：治せた。でも“多数決9回”には届かなかった")
    pic_fit(s, str(AST / "fig_verdict.png"), 0.85, 1.85, 8.0, 3.6, align="center", valign="top")
    # 右に要点
    rect(s, 9.15, 2.0, 3.3, 3.4, fill=PAPER, line=LINE, line_w=1.1,
         shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06)
    para_block(s, 9.4, 2.2, 2.85, 3.1, [
        [("要点", 12.5, ACCENT, FONT_SB, {})],
        [("○ 処方で+3.2点の改善", 13.5, POS, FONT, {"bold": True})],
        [("○ ベースと互角まで回復", 13, INK, FONT, {})],
        [("× 多数決9回には−3.0点", 13.5, NEG, FONT, {"bold": True})],
        [("＝“工夫を尽くしても素朴な強敵に勝てなかった”という結果", 12.5, SUB, FONT, {})],
    ], space_after=11, line_spacing=1.25)
    rect(s, 0.85, 5.75, 11.5, 1.1, fill=INK_PANEL, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.05)
    text(s, 1.2, 5.95, 10.9, 0.75, [
        ("これは“ダメだった”ではなく、", 15.5, CARD, FONT, {}),
        ("「条件をとことん公平にすると議論の優位は消える」という批判を、"
         "最も丁寧に確かめた結果", 15.5, RGBColor(0xF0,0xD9,0xC7), FONT, {"bold": True}),
        ("。", 15.5, CARD, FONT, {})], line_spacing=1.25)
    foot(s)
    notes(s, "そして正直な結論です。処方でプラス3.2点の改善はできて、ベースと互角まで回復した。"
             "でも、多数決9回にはマイナス3.0点で、届きませんでした。"
             "工夫を尽くしても、素朴な強敵には勝てなかった、という結果です。"
             "ただ、これはダメだったで終わる話ではありません。"
             "条件をとことん公平にすると議論の優位は消える、という批判が前からあったんですが、"
             "それを最も丁寧に、いろんな対策を打った上で確かめた結果なんです。"
             "négatifな結果も、きちんと確かめれば立派な成果だと考えています。")


# ============================================================
# 章扉E：まとめ
# ============================================================
def div_wrap():
    s = section_divider(prs, 5, "考察・結論・今後",
                        "なぜこうなったのか。何が言えて、次に何をするか。")
    notes(s, "最後に、考察と結論、そして今後です。なぜこうなったのか、何が言えて、次に何をするかを話します。")


def sl_discuss_why():
    s = add_slide(prs)
    title_head(s, "考察①", "なぜ勝てないのか？ でも、どこに価値があるのか？")
    pic_fit(s, str(AST / "fig_budget.png"), 0.85, 1.75, 7.2, 3.6, align="center", valign="top")
    para_block(s, 8.2, 2.0, 4.2, 3.6, [
        [("多数決9回は、", 14.5, INK, FONT, {}),
         ("独立した9票", 14.5, SUB, FONT, {"bold": True}),
         ("の“数の力”が強い。", 14.5, INK, FONT, {})],
        [("チームは実質3つの視点しかなく、"
          "議論でその差を埋めきれなかった。", 14, INK, FONT, {})],
        [("ただし", 14.5, INK, FONT, {}),
         ("数学では互角", 14.5, POS, FONT, {"bold": True}),
         ("。確かめられる問題では、少ない視点でも議論が補える。", 14, INK, FONT, {})],
    ], space_after=14, line_spacing=1.28)
    rect(s, 0.85, 5.7, 11.5, 1.05, fill=PRAG_BG, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.05)
    text(s, 1.2, 5.88, 10.9, 0.7, [
        ("要するに——", 14.5, PRAG, FONT, {"bold": True}),
        ("“数で押す”のが有利な問題と、“話し合い”が活きる問題は別。使い分けが本質。",
         14.5, INK, FONT, {})], line_spacing=1.2)
    foot(s)
    notes(s, "なぜ勝てないのか。多数決9回は、独立した9票の数の力が強いんです。"
             "一方チームは、3体×2ラウンドとはいえ、実質3つの視点しかない。議論でその差を埋めきれなかった。"
             "ただし数学では互角でした。確かめられる問題では、視点が少なくても議論が補えるんです。"
             "要するに、数で押すのが有利な問題と、話し合いが活きる問題は別で、使い分けが本質だ、ということです。")


def sl_discuss_evo():
    s = add_slide(prs)
    title_head(s, "考察②", "2つの“気づき” —— 進化の正体と、評価の落とし穴")
    cards = [
        ("進化は「選抜」ではなく「修復」だった", TEAL,
         "世代を回すと重みの“向き”だけが回転し、性格を教えたときの"
         "傷が打ち消されていた。つまり進化＝壊れた能力の回復。"
         "ただし同じ回復は、学び直し（リプレイ）の方が安く確実にできた。"),
        ("同じ設定でも“測る環境”で点数が動く", ACCENT,
         "評価プログラムを入れ替えただけで、同じ問題・同じ設定なのに"
         "6点ほどズレる現象を発見。しかもベース側だけ過小評価されやすい。"
         "→ 比較は必ず同じ環境で。ここは方法論としての教訓。"),
    ]
    y = 2.05
    for hd, col, body in cards:
        rect(s, 0.85, y, 11.5, 2.05, fill=CARD, line=LINE, line_w=1.0,
             shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.05, shadow=True)
        rect(s, 0.85, y, 0.14, 2.05, fill=col, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.5)
        text(s, 1.2, y + 0.2, 11.0, 0.5, [(hd, 16.5, col, FONT, {"bold": True})])
        text(s, 1.2, y + 0.78, 10.9, 1.1, [(body, 14, INK, FONT, {})], line_spacing=1.3)
        y += 2.25
    foot(s)
    notes(s, "考察の2つ目。研究の途中で得た、大事な気づきが2つあります。"
             "1つ目。進化は、当初ねらった“良い個体の選抜”ではなく、実は“修復”でした。"
             "世代を回すと重みの向きだけが回転して、性格を教えたときの傷が打ち消されていたんです。"
             "ただ、同じ回復は学び直しの方が安く確実にできたので、そちらを採用しました。"
             "2つ目。同じ問題・同じ設定なのに、評価プログラムを入れ替えただけで6点ほどズレる現象を見つけました。"
             "しかもベース側だけ過小評価されやすい。比較は必ず同じ環境で、というのは方法論としての教訓です。"
             "危うく間違った結論を出すところでした。")


def sl_conclusion():
    s = add_slide(prs)
    title_head(s, "結論", "この研究で分かったこと")
    items = [
        ("チームレベルの貢献度で進化させる枠組みを作り、実際に動かした",
         "3体なら貢献度を近似なしで公平に測れる。仕組みとして初めて形にした。"),
        ("議論の効き方は「領域」で決まる",
         "確かめられる問題では効き、そうでない問題では逆効果。統一的に説明できた。"),
        ("失敗の原因を分解し、処方で“治せる”ことを示した",
         "能力の毀損を特定し、学び直しでベース水準まで回復させた（+3.2点）。"),
        ("それでも“多数決9回”には勝てない、を厳密に確認（正直な負け）",
         "条件を公平にし、環境の罠も統制した上での結論。数学だけは互角。"),
    ]
    y = 1.95
    for i, (hd, body) in enumerate(items):
        rect(s, 0.85, y, 11.5, 1.12, fill=CARD, line=LINE, line_w=1.0,
             shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.05, shadow=True)
        # 番号丸
        rect(s, 1.05, y + 0.28, 0.56, 0.56, fill=TEAL, shape=MSO_SHAPE.OVAL)
        text(s, 1.05, y + 0.28, 0.56, 0.56, [(str(i+1), 17, CARD, FONT, {"bold": True})],
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        text(s, 1.85, y + 0.16, 10.3, 0.5, [(hd, 15, INK, FONT, {"bold": True})])
        text(s, 1.85, y + 0.62, 10.3, 0.45, [(body, 12.5, SUB, FONT, {})])
        y += 1.24
    foot(s)
    notes(s, "結論です。4点にまとめました。"
             "1つ目。チームへの貢献度で進化させる枠組みを作って、実際に動かしました。"
             "3体なら貢献度を近似なしで公平に測れる、という点を初めて形にしています。"
             "2つ目。議論の効き方は領域で決まる。確かめられる問題では効いて、そうでないと逆効果、と統一的に説明できました。"
             "3つ目。失敗の原因を分解して、処方で治せることを示した。能力の毀損を特定して、学び直しでベース水準まで回復させました。"
             "4つ目。それでも多数決9回には勝てない、を厳密に確認した。これは正直な負けですが、"
             "条件を公平にし、環境の罠も統制した上での、信頼できる結論です。数学だけは互角でした。")


def sl_future():
    s = add_slide(prs)
    title_head(s, "今後の展開", "次にやること")
    items = [
        ("“いいとこ取り”を試す", "各メンバーが数回ずつ解いてから議論する"
         "（数の力＋話し合いの合わせ技）。今回の結果が指す最有力の次の一手。"),
        ("進化のやり方を鍛え直す", "選抜のノイズを抑える改良版で、"
         "進化が“修復”を超えて“発見”になるかを検証する。"),
        ("違う性格・違うモデルを混ぜる", "似た者同士ではなく、"
         "本当に異質なメンバーを組ませて多様性を最大化する。"),
    ]
    y = 2.05
    for hd, body in items:
        rect(s, 0.85, y, 11.5, 1.35, fill=CARD, line=LINE, line_w=1.0,
             shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.05, shadow=True)
        rect(s, 0.85, y, 0.14, 1.35, fill=ACCENT, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.5)
        text(s, 1.2, y + 0.2, 11.0, 0.5, [("▶ " + hd, 16, INK, FONT, {"bold": True})])
        text(s, 1.5, y + 0.72, 10.6, 0.5, [(body, 13.5, SUB, FONT, {})], line_spacing=1.2)
        y += 1.5
    foot(s)
    notes(s, "今後の展開です。3つ考えています。"
             "1つ目。いいとこ取りを試す。各メンバーが数回ずつ解いてから議論する、"
             "数の力と話し合いの合わせ技です。今回の結果が指す、最有力の次の一手だと思っています。"
             "2つ目。進化のやり方を鍛え直す。選抜のノイズを抑えた改良版で、"
             "進化が修復を超えて、本当の発見になるかを検証します。"
             "3つ目。違う性格や違うモデルを混ぜる。似た者同士ではなく、本当に異質なメンバーを組ませて、多様性を最大化したい。")


def sl_closing():
    s = add_slide(prs, bg=INK_PANEL)
    rect(s, 0, 0, 0.28, 7.5, fill=ACCENT)
    text(s, 1.2, 2.15, 11.0, 1.0, [("まとめ", 15, ACCENT, FONT_SB, {})])
    para_block(s, 1.2, 2.7, 10.8, 3.0, [
        [("小さなAIを3体、話し合わせて進化させた。", 26, CARD, FONT, {"bold": True})],
        [("議論には“効く問題”と“効かない問題”があり、", 19, RGBColor(0xE7,0xEC,0xEF), FONT, {})],
        [("失敗は診断でき、処方で治せた。", 19, RGBColor(0xE7,0xEC,0xEF), FONT, {})],
        [("——ただし、素朴な強敵に勝つのはこれからの宿題。", 18, RGBColor(0xF0,0xD9,0xC7), FONT, {})],
    ], space_after=14, line_spacing=1.2)
    line(s, 1.25, 6.1, 4.0, 0, color=ACCENT, weight=1.5)
    text(s, 1.2, 6.3, 11.0, 0.5, [("ご清聴ありがとうございました。", 15, CARD, FONT, {})])
    notes(s, "まとめます。小さなAIを3体、話し合わせて進化させました。"
             "議論には効く問題と効かない問題があって、失敗は診断でき、処方で治せた。"
             "ただし、素朴な強敵に勝つのは、これからの宿題です。"
             "以上で発表を終わります。ご清聴ありがとうございました。ご質問をお願いします。")


# ============================================================
# 付録
# ============================================================
def div_appendix():
    s = section_divider(prs, 6, "付録", "数値の詳細・用語・再現性")
    notes(s, "以降は付録です。質疑に備えて、数値の詳細や用語の説明を用意しています。")


def sl_apx_numbers():
    s = add_slide(prs)
    title_head(s, "付録 A1", "最終結果の数値表（正答率 ％・測定環境を統一）")
    rows = [
        ("条件", "一般知識", "数学", "科学", "備考"),
        ("ベース単体", "72.7", "81.8", "43.1", "1体で1回"),
        ("多数決9回（SC@9）", "74.0", "87.3", "48.6", "最強の比較相手"),
        ("旧チーム（処方前）", "68.5", "79.3", "42.0", "ベース以下に低下"),
        ("新チーム（処方後）", "71.6", "86.7", "43.1", "ベースと互角に回復"),
    ]
    x0, y0 = 0.95, 2.1
    ws = [3.4, 1.9, 1.7, 1.7, 2.6]
    rh = 0.62
    for r, row in enumerate(rows):
        x = x0
        for c, cell in enumerate(row):
            head = (r == 0)
            fill = INK_PANEL if head else (CARD if r % 2 else PAPER)
            rect(s, x, y0 + r * rh, ws[c], rh, fill=fill, line=LINE, line_w=0.8)
            tcol = CARD if head else INK
            bold = head or c == 0
            emph = (r == 4)  # 新チームを強調
            if emph and not head:
                tcol = TEAL if c > 0 else INK
            text(s, x + 0.12, y0 + r * rh, ws[c] - 0.24, rh,
                 [(cell, 12.5 if head else 12, tcol, FONT_SB if bold else FONT, {})],
                 align=PP_ALIGN.LEFT if c in (0, 4) else PP_ALIGN.CENTER,
                 anchor=MSO_ANCHOR.MIDDLE)
            x += ws[c]
    text(s, 0.95, 5.5, 11.4, 1.2, [
        ("・数学（MATH-500）では新チームが多数決9回と統計的に互角（差 −0.7点, 有意差なし）。\n", 12.5, INK, FONT, {}),
        ("・総合（6,000問）では新チームが多数決9回に −3.0点（有意）。ベース単体とは互角（+0.3点）。\n", 12.5, INK, FONT, {}),
        ("・多数決9回とベース／新チームは6シード、旧チームは3シードで測定。", 12, SUB, FONT, {}),
    ], line_spacing=1.3)
    apx_foot(s, 1)
    notes(s, "付録の数値表です。質疑用です。"
             "測定環境をそろえた最終値で、新チームはベースと互角、数学では多数決9回とも互角です。"
             "ただ総合では多数決9回に3点届かない。ここが正直なところです。")


def sl_apx_persona():
    s = add_slide(prs)
    title_head(s, "付録 A2", "3つの性格に与えた実際の指示文")
    data = [
        ("批判", CRITIC, CRITIC_BG,
         "あなたは厳密な検証を重視する批判的思考家。反証・例外・境界条件に敏感。"),
        ("実務", PRAG, PRAG_BG,
         "あなたは応用志向の実務家。意思決定に役立つ実装可能性とコストを重視。"),
        ("探索", EXPLORE, EXPLORE_BG,
         "あなたは創発を促す発想家。仮説生成と多角的比喩で発想を広げる。"),
    ]
    y = 2.15
    for nm, col, bg, txt in data:
        rect(s, 0.85, y, 11.5, 1.15, fill=bg, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06)
        rect(s, 1.1, y + 0.34, 1.5, 0.48, fill=col, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.4)
        text(s, 1.1, y + 0.34, 1.5, 0.48, [(nm, 14, CARD, FONT_SB, {})],
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        text(s, 2.9, y + 0.2, 9.2, 0.8, [("「" + txt + "」", 15, INK, FONT, {})],
             anchor=MSO_ANCHOR.MIDDLE, line_spacing=1.2)
        y += 1.33
    text(s, 0.85, 6.35, 11.6, 0.5,
         [("たった1文ずつ。これだけで、デモで見たような異なる解き方が生まれます。", 13.5, SUB, FONT, {})])
    apx_foot(s, 2)
    notes(s, "3つの性格に与えた実際の指示文です。それぞれたった1文ずつ。"
             "この短い指示だけで、デモで見たような異なる解き方が生まれる、というのは面白い点だと思います。")


def sl_apx_terms():
    s = add_slide(prs)
    title_head(s, "付録 A3", "用語ミニ辞典")
    terms = [
        ("LoRA", "土台モデルに足す小さな“追加重み”。少ない容量で性格や技能を足せる。"),
        ("SC@9 / 多数決9回", "同じAIに9回バラバラに解かせ、多数決を取る素朴で強い方法。"),
        ("Shapley値（貢献度）", "「その人が抜けたらどれだけ困るか」を全組み合わせで公平に測った値。"),
        ("MMLU-Pro / MATH-500 / SuperGPQA", "順に、一般知識・推論／数学／大学院級科学の標準テスト。"),
        ("進化（交叉・突然変異）", "良い個体の重みを掛け合わせ・少し変えて次世代を作る最適化。"),
    ]
    y = 2.0
    for t, d in terms:
        rect(s, 0.85, y, 3.5, 0.82, fill=INK_PANEL, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08)
        text(s, 1.05, y, 3.1, 0.82, [(t, 13, CARD, FONT_SB, {})],
             anchor=MSO_ANCHOR.MIDDLE, line_spacing=1.05)
        rect(s, 4.5, y, 7.85, 0.82, fill=CARD, line=LINE, line_w=0.9,
             shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08)
        text(s, 4.75, y, 7.4, 0.82, [(d, 13, INK, FONT, {})],
             anchor=MSO_ANCHOR.MIDDLE, line_spacing=1.15)
        y += 0.95
    apx_foot(s, 3)
    notes(s, "最後に用語のミニ辞典です。LoRA、多数決9回、シャープレイ値、各テスト、進化。"
             "本編で噛み砕いた言葉の、正式な対応関係をまとめています。質疑の参考にどうぞ。")


def build_all():
    # 背景
    cover(); sl_intro()
    div_background(); sl_bg_scale(); sl_bg_debate(); sl_bg_critique(); sl_bg_gap()
    # 手法
    div_method(); sl_method_overview(); sl_personas(); sl_shapley(); sl_evolution()
    # デモ
    div_demo(); sl_demo_intro()
    sl_cantor_q(); sl_cantor_r1(); sl_cantor_r2(); sl_cantor_msg()
    sl_crypto_q(); sl_crypto_r2(); sl_crypto_msg()
    # 実験と結果
    div_results(); sl_setup(); sl_finding_domain(); sl_finding_repair()
    sl_final_1(); sl_final_2()
    # 考察・結論・今後
    div_wrap(); sl_discuss_why(); sl_discuss_evo(); sl_conclusion(); sl_future(); sl_closing()
    # 付録
    div_appendix(); sl_apx_numbers(); sl_apx_persona(); sl_apx_terms()


if __name__ == "__main__":
    build_all()
    out = HERE / "研究進捗_20260709.pptx"
    prs.save(str(out))
    print("saved", out, "slides:", len(prs.slides._sldIdLst), "pages:", _page["n"])
