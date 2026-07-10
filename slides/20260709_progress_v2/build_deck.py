# -*- coding: utf-8 -*-
"""研究進捗スライド v2（2026-07-09 報告）。源暎エムゴv2・ウェイト使い分け・進化を正直に。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from deck_helpers import *  # noqa

HERE = Path(__file__).parent
AST = HERE / "assets"
prs = new_deck()

TOTAL = 27
_page = {"n": 0}


def foot(slide):
    _page["n"] += 1
    footer(slide, _page["n"], TOTAL)


def apx_foot(slide, idx, total=3):
    text(slide, 0.85, 7.06, 8.0, 0.3,
         [("LoRAエージェント集団の進化 ／ 研究進捗", 9.5, FAINT, FONT_REG, {})],
         anchor=MSO_ANCHOR.MIDDLE)
    text(slide, 11.2, 7.06, 1.3, 0.3, [(f"付録 A{idx} / {total}", 9.5, FAINT, FONT_REG, {})],
         align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE)


def cite(slide, x, y, w, txt):
    text(slide, x, y, w, 0.34, [("出典：" + txt, 10.5, FAINT, FONT_REG, {})],
         anchor=MSO_ANCHOR.MIDDLE)


# ============================================================
# 表紙
# ============================================================
def cover():
    s = add_slide(prs, bg=PAPER)
    rect(s, 0, 0, 13.333, 0.28, fill=INK_PANEL)
    rect(s, 0, 7.22, 13.333, 0.28, fill=ACCENT)
    text(s, 0.95, 1.15, 11.4, 0.5, [("修士研究 進捗報告", 15, ACCENT, FONT_BOLD, {})])
    text(s, 0.9, 1.75, 11.6, 2.2,
         [("小さなAIを3体、話し合わせて鍛える", 34, INK, FONT_BLACK, {})], line_spacing=1.15)
    text(s, 0.92, 2.66, 11.6, 1.4,
         [("——「議論するAIチーム」は、素朴なやり方に勝てるのか？", 19, SUB, FONT_MED, {})])
    text(s, 0.95, 3.72, 11.5, 0.9, [
        ("Evolutionary Optimization of LoRA Agent Populations", 14, FAINT, FONT_REG, {}),
        ("\nwith Team-Level Fitness in Multi-Agent Debate", 14, FAINT, FONT_REG, {}),
    ], line_spacing=1.2)
    line(s, 0.95, 5.25, 5.3, 0, color=LINE, weight=1.3)
    para_block(s, 0.95, 5.5, 11.0, 1.4, [
        [("新潟大学 自然科学研究科　電気情報工学専攻　情報社会デザイン科学コース", 14, INK, FONT_REG, {})],
        [("舛田 岳　（学籍番号 F25C142E）", 15, INK, FONT_BOLD, {})],
        [("2026年7月9日", 13, SUB, FONT_REG, {})],
    ], space_after=6)
    notes(s, "本日はよろしくお願いします。修士研究の進捗を報告します。"
             "テーマは、小さなAIを3体用意して話し合わせ、チームとして鍛えていく、というものです。"
             "今日いちばん見ていただきたいのは、実際にAIたちがどう会話しているか、その生のログです。"
             "結論から言うと、良かった点も、正直うまくいかなかった点もあります。そこも隠さずお話しします。")


# ============================================================
# 1. ワンライナー
# ============================================================
def sl_intro():
    s = add_slide(prs)
    title_head(s, "この研究を一言でいうと", "「1体の賢いAI」より「3体で話し合うAI」は強いのか？")
    cards = [
        ("やったこと", TEAL, [
            "同じ小型AIに3つの“性格”を持たせる",
            "互いの答えを見せ合って議論させる",
            "チームとして賢く・頑丈になるよう鍛える",
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
        rect(s, x, 2.45, w, 0.32, fill=col)
        text(s, x, 2.15, w, 0.62, [(hd, 16, CARD, FONT_BOLD, {})],
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        bullets(s, x + 0.32, 3.02, w - 0.6, 2.7, items, size=13.5, gap=12,
                marker="●", marker_color=col)
    text(s, 0.85, 6.15, 11.6, 0.7,
         [("キーワードは ", 15, INK, FONT_REG, {}),
          ("「多様性」と「協調」", 15, ACCENT, FONT_BOLD, {}),
          ("。ただし、いいことばかりではありませんでした。", 15, INK, FONT_REG, {})])
    foot(s)
    notes(s, "まず全体像です。やったのは大きく3つ。1つの土台モデルに3つの性格を持たせ、"
             "互いの答えを見せ合って議論させ、チームとして賢く頑丈になるよう鍛える、というものです。"
             "調べたのは、こうした議論チームが、単純に何回も解いて多数決するやり方に勝てるのか。"
             "どんな問題で効いて、どんな問題でダメなのか。そしてダメなときの原因はどこか。"
             "今日は本物の会話ログで、うまくいく例と崩れる例の両方をお見せします。"
             "キーワードは多様性と協調ですが、正直いいことばかりではなかった、というのが裏テーマです。")


# ============================================================
# 章扉① 背景と問い
# ============================================================
def div_background():
    s = section_divider(prs, 1, "背景と問い",
                        "なぜ「小さなAIを話し合わせる」のか。先行研究は何と言っているか。")
    notes(s, "まず背景です。なぜ小さなAIをわざわざ話し合わせるのか、"
             "そして世の中の研究がこれについて何と言っているかを、かいつまんで説明します。")


def sl_bg_scale():
    s = add_slide(prs)
    title_head(s, "背景①", "モデルは「大きくすれば勝ち」なのか？")
    para_block(s, 0.85, 1.95, 7.0, 3.6, [
        [("これまでのAIは、", 17, INK, FONT_REG, {}),
         ("大きくするほど賢くなってきました", 17, INK, FONT_BOLD, {}),
         ("。", 17, INK, FONT_REG, {})],
        [("でも大きなモデルは、動かすのにお金も電力もかかる。"
          "際限なく大きくする道には限界があります。", 16, INK, FONT_REG, {})],
        [("そこで別の方向として、", 16, INK, FONT_REG, {}),
         ("小さなモデルを何体か集めて協力させる", 16, ACCENT, FONT_BOLD, {}),
         ("考え方が注目されています。", 16, INK, FONT_REG, {})],
        [("実際、同じAIに何度も解かせて多数決するだけでも、"
          "体数を増やすほど成績が上がると報告されています。", 15, SUB, FONT_REG, {})],
    ], space_after=13)
    cite(s, 0.85, 5.7, 7.0, "Liら（2024）「More Agents Is All You Need」")
    rect(s, 8.4, 2.15, 3.9, 1.5, fill=CARD, line=LINE, line_w=1.1,
         shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06, shadow=True)
    text(s, 8.4, 2.28, 3.9, 0.5, [("これまで", 12, SUB, FONT_BOLD, {})], align=PP_ALIGN.CENTER)
    text(s, 8.4, 2.72, 3.9, 0.8, [("🧠 1体の巨大モデル", 17, INK, FONT_BOLD, {})],
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    text(s, 8.4, 3.9, 3.9, 0.5, [("↓", 22, ACCENT, FONT_BOLD, {})], align=PP_ALIGN.CENTER)
    rect(s, 8.4, 4.35, 3.9, 1.7, fill=TEAL, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
         radius=0.06, shadow=True)
    text(s, 8.4, 4.5, 3.9, 0.5, [("この研究", 12, RGBColor(0xD9,0xE6,0xEC), FONT_BOLD, {})],
         align=PP_ALIGN.CENTER)
    text(s, 8.4, 4.95, 3.9, 0.9, [("🤝 小さなAI×3体\nで話し合う", 16, CARD, FONT_BOLD, {})],
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    foot(s)
    notes(s, "AIはこれまで、大きくするほど賢くなる、という流れで発展してきました。"
             "でも大きなモデルはお金も電力もかかって、無限には大きくできません。"
             "そこで注目されているのが、小さなモデルを何体か集めて協力させる方向です。"
             "実際、2024年のLiらの研究では、同じAIに何度も解かせて多数決するだけでも、"
             "体数を増やすほど成績が上がると報告されています。数の力ですね。"
             "この研究はまさに、小さなAIを3体話し合わせる立場に立っています。")


def sl_bg_debate():
    s = add_slide(prs)
    title_head(s, "背景②", "「AI同士で議論させると賢くなる」という報告")
    rect(s, 0.85, 1.95, 5.65, 3.9, fill=CRITIC_BG, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.05)
    rect(s, 0.85, 1.95, 0.13, 3.9, fill=CRITIC, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.5)
    text(s, 1.2, 2.2, 5.0, 0.5, [("Duら（2023）", 14, CRITIC, FONT_BOLD, {})])
    para_block(s, 1.2, 2.75, 5.05, 3.0, [
        [("3体のAIが、互いの答えを見せ合いながら"
          "2ラウンド考え直すと精度が上がった。", 14.5, INK, FONT_REG, {})],
        [("算数テスト（GSM8K）", 13.5, INK, FONT_MED, {}),
         ("　77 → 85点", 15, POS, FONT_BOLD, {})],
        [("知識テスト（MMLU）", 13.5, INK, FONT_MED, {}),
         ("　64 → 71点", 15, POS, FONT_BOLD, {})],
        [("＝「一人で悩むより、視点の違う相手と"
          "話す方がよい」という直感に合う。", 13.5, SUB, FONT_REG, {})],
    ], space_after=11, line_spacing=1.25)
    rect(s, 6.7, 1.95, 5.65, 3.9, fill=EXPLORE_BG, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.05)
    rect(s, 6.7, 1.95, 0.13, 3.9, fill=EXPLORE, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.5)
    text(s, 7.05, 2.2, 5.0, 0.5, [("Liangら（2023）", 14, EXPLORE, FONT_BOLD, {})])
    para_block(s, 7.05, 2.75, 5.05, 3.0, [
        [("一人で反省すると、同じ考えに"
          "こだわり続けてしまう（“思考の退化”）。", 14.5, INK, FONT_REG, {})],
        [("違う視点の相手とぶつけ合う議論が、"
          "その行き詰まりから抜け出させる。", 14.5, INK, FONT_REG, {})],
        [("＝多様な視点をぶつけることに"
          "価値がある、という主張。", 13.5, SUB, FONT_REG, {})],
    ], space_after=12, line_spacing=1.3)
    foot(s)
    notes(s, "その協力のやり方で有名なのが議論です。2023年のDuらの研究では、"
             "3体のAIが互いの答えを見せ合って2ラウンド考え直すと、算数テストが77から85点、"
             "知識テストが64から71点に上がった、と報告されました。"
             "一人で悩むより視点の違う相手と話す方がいい。人間の感覚にも合いますよね。"
             "同じ2023年のLiangらは、一人で反省すると同じ考えにこだわってしまう、これを思考の退化と呼び、"
             "違う視点の議論が抜け出させる、と言っています。多様な視点をぶつける価値、ですね。")


def sl_bg_critique():
    s = add_slide(prs)
    title_head(s, "背景③", "——ところが「そんなに良くない」という反論も強い")
    items = [
        ("計算量をそろえると勝てない", NEG,
         "議論は何度もAIを呼ぶ。同じ手間を“ただ何回も解いて多数決”に使うと、そちらが強い。",
         "Smitら（ICML 2024）"),
        ("相手に流されて間違える", GOLD,
         "自信のある正解でも、多数派の空気に合わせて誤答に書き換えてしまう（“追従”）。",
         "「Talk isn’t Cheap」（2025）"),
        ("評価を正すと優位が消える／鍵は“異質性”", SUB,
         "評価の不備を正すと議論の優位は消える。効かせる鍵はメンバーの多様性・異質性だと指摘。",
         "Zhangら（2025）"),
    ]
    y = 2.0
    for hd, col, body, src in items:
        rect(s, 0.85, y, 11.5, 1.32, fill=CARD, line=LINE, line_w=1.0,
             shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.05, shadow=True)
        rect(s, 0.85, y, 0.13, 1.32, fill=col, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.5)
        text(s, 1.2, y + 0.15, 8.6, 0.5, [(hd, 16, col, FONT_BOLD, {})])
        text(s, 1.2, y + 0.64, 9.3, 0.6, [(body, 13, INK, FONT_REG, {})], line_spacing=1.15)
        text(s, 10.0, y + 0.15, 2.25, 1.0, [(src, 11, FAINT, FONT_REG, {})],
             anchor=MSO_ANCHOR.MIDDLE, line_spacing=1.15)
        y += 1.46
    text(s, 0.85, 6.5, 11.6, 0.5,
         [("→ とくに ", 13.5, INK, FONT_REG, {}),
          ("小さいモデルでは、素朴な議論はうまくいかない可能性が高い", 13.5, NEG, FONT_BOLD, {}),
          ("。ここが出発点。", 13.5, INK, FONT_REG, {})])
    foot(s)
    notes(s, "ところが、そう単純じゃない、という反論も強いんです。3つ挙げます。"
             "1つ目、SmitらのICML2024の研究。議論は何度もAIを呼ぶ手間がかかる。"
             "その同じ手間をただ何回も解いて多数決に使うと、そっちの方が強い、という指摘。これが一番手強い。"
             "2つ目、Talk isn’t Cheapという2025年の研究。自信のある正解でも多数派に流されて誤答に書き換える、追従です。"
             "3つ目、Zhangらの2025年の研究。評価の不備を正すと議論の優位は消える。効かせる鍵はメンバーの異質性だと。"
             "とくに小さいモデルでは、素朴な議論はうまくいかない可能性が高い。ここから私の研究は出発します。")


def sl_bg_gap():
    s = add_slide(prs)
    title_head(s, "背景④・本研究の問い", "既存研究は「メンバーを固定」して議論のやり方だけ工夫してきた")
    para_block(s, 0.85, 1.95, 11.4, 1.5, [
        [("これまでの改善は、", 16, INK, FONT_REG, {}),
         ("ラウンド数や集約方法など“議論の進め方”をいじる", 16, INK, FONT_BOLD, {}),
         ("ものが中心でした。", 16, INK, FONT_REG, {})],
        [("でも、議論の良し悪しは", 16, INK, FONT_REG, {}),
         ("参加するメンバー自身の質と多様性", 16, ACCENT, FONT_BOLD, {}),
         ("に強く左右されるはず。だったら——", 16, INK, FONT_REG, {})],
    ], space_after=12)
    rect(s, 0.85, 4.0, 11.5, 2.1, fill=TEAL, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
         radius=0.05, shadow=True)
    text(s, 1.3, 4.35, 10.7, 1.6, [
        ("本研究の問い", 14, RGBColor(0xCF,0xE1,0xE9), FONT_BOLD, {}),
        ("\nメンバー（AI）自身を、"
         "“チームへの貢献度”を手がかりに鍛えられないか？", 21, CARD, FONT_BOLD, {}),
    ], line_spacing=1.25)
    foot(s)
    notes(s, "ここが問いの核心です。これまでの改善は、ラウンド数を増やすとか集約を変えるとか、"
             "議論の進め方をいじるものが中心でした。メンバー自身は固定していたんです。"
             "でも議論の良し悪しって、結局は参加するメンバーの質と多様性で決まるはずですよね。"
             "だったら、メンバーであるAI自身を、チームへの貢献度を手がかりに鍛えられないか。これが本研究の問いです。")


# ============================================================
# 章扉② 提案手法
# ============================================================
def div_method():
    s = section_divider(prs, 2, "提案手法",
                        "3つの性格をどう作り、どう議論させ、どう鍛えるか。")
    notes(s, "では具体的にどういう仕組みなのかを説明します。"
             "3つの性格をどう作って、どう議論させて、どう鍛えるか、です。")


def sl_method_overview():
    s = add_slide(prs)
    title_head(s, "手法①", "全体像：1回の議論と、チームの鍛え方")
    text(s, 0.85, 1.7, 11.6, 0.4, [("① 1回の議論（本番で問題を解くとき）", 14, TEAL, FONT_BOLD, {})])
    rect(s, 0.9, 2.35, 1.95, 1.35, fill=INK_PANEL, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
         radius=0.08, shadow=True)
    text(s, 0.9, 2.5, 1.95, 1.05, [("土台モデル", 13, CARD, FONT_BOLD, {}),
         ("\nQwen3-4B\n（共通の1体）", 11, RGBColor(0xC7,0xCE,0xD4), FONT_REG, {})],
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, line_spacing=1.15)
    roles = [("批判", CRITIC, CRITIC_BG), ("実務", PRAG, PRAG_BG), ("探索", EXPLORE, EXPLORE_BG)]
    for i, (nm, col, bg) in enumerate(roles):
        yy = 2.18 + i * 0.56
        rect(s, 3.55, yy, 1.5, 0.46, fill=bg, line=col, line_w=1.2,
             shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.4)
        text(s, 3.55, yy, 1.5, 0.46, [(f"性格：{nm}", 12, col, FONT_BOLD, {})],
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    line(s, 2.9, 3.02, 0.6, 0, color=SUB, weight=1.3)
    rect(s, 5.5, 2.35, 1.85, 1.35, fill=CARD, line=TEAL, line_w=1.4,
         shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08, shadow=True)
    text(s, 5.5, 2.5, 1.85, 1.05, [("議論", 14, TEAL, FONT_BOLD, {}),
         ("\n答えを見せ合い\n2ラウンド", 11, INK, FONT_REG, {})],
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, line_spacing=1.15)
    text(s, 5.05, 2.85, 0.5, 0.5, [("→", 20, SUB, FONT_BOLD, {})], align=PP_ALIGN.CENTER)
    rect(s, 8.0, 2.35, 1.85, 1.35, fill=CARD, line=LINE, line_w=1.2,
         shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08, shadow=True)
    text(s, 8.0, 2.5, 1.85, 1.05, [("答えを集約", 13, INK, FONT_BOLD, {}),
         ("\n重み付き投票\nで最終回答", 11, SUB, FONT_REG, {})],
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, line_spacing=1.15)
    text(s, 7.45, 2.85, 0.5, 0.5, [("→", 20, SUB, FONT_BOLD, {})], align=PP_ALIGN.CENTER)
    text(s, 10.05, 2.85, 2.2, 0.5, [("→ 最終回答", 15, INK, FONT_BOLD, {})],
         anchor=MSO_ANCHOR.MIDDLE)
    line(s, 0.85, 4.35, 11.6, 0, color=LINE, weight=1.0)
    text(s, 0.85, 4.5, 11.6, 0.4, [("② チームの鍛え方（本番の前に）", 14, ACCENT, FONT_BOLD, {})])
    steps = [("性格を持たせる\n（3つのLoRA）", INK_PANEL, CARD),
             ("賢さを保つ\n（リプレイ＝復習）", TEAL, CARD),
             ("議論を頑丈に\n（匿名化など）", ACCENT, CARD),
             ("貢献度で選ぶ\n（進化）", POS, CARD)]
    x = 0.9
    for i, (tx, col, tc) in enumerate(steps):
        rect(s, x, 5.05, 2.5, 1.1, fill=col, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
             radius=0.08, shadow=True)
        text(s, x, 5.05, 2.5, 1.1, [(tx, 12.5, tc, FONT_BOLD, {})],
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, line_spacing=1.15)
        if i < 3:
            text(s, x + 2.5, 5.3, 0.42, 0.6, [("→", 18, SUB, FONT_BOLD, {})],
                 align=PP_ALIGN.CENTER)
        x += 2.92
    text(s, 0.9, 6.35, 11.5, 0.5, [("この後、②の各ステップを順に説明します。", 13, SUB, FONT_REG, {})])
    foot(s)
    notes(s, "全体像です。上下2段に分けました。"
             "上段が本番、1回の議論。共通の土台モデル1体に、批判・実務・探索の3つの性格を持たせ、"
             "3体が同じ問題を解いて2ラウンド議論し、最後に重み付き投票で結論を出す。"
             "下段が、本番の前にチームをどう鍛えるか。性格を持たせ、賢さを保ち、議論を頑丈にし、"
             "貢献度で選ぶ。この4ステップを、このあと順に説明していきます。")


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
        text(s, x, 2.24, w, 0.7, [(nm, 16.5, CARD, FONT_BOLD, {})],
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        text(s, x + 0.3, 3.2, w - 0.6, 0.5, [(tag, 13.5, col, FONT_BOLD, {})],
             align=PP_ALIGN.CENTER)
        line(s, x + 0.5, 3.72, w - 1.0, 0, color=LINE, weight=1.0)
        text(s, x + 0.32, 3.95, w - 0.62, 1.5, [(role, 13, INK, FONT_REG, {})],
             align=PP_ALIGN.CENTER, line_spacing=1.25)
    text(s, 0.85, 5.95, 11.6, 0.9,
         [("実際には、これらは短い日本語の指示文で与えます（付録に原文）。"
           "大事なのは ", 13.5, INK, FONT_REG, {}),
          ("“同じ問題を、あえて違う切り口で見る3体”", 13.5, ACCENT, FONT_BOLD, {}),
          ("を用意したこと。", 13.5, INK, FONT_REG, {})])
    foot(s)
    notes(s, "3つの性格を紹介します。批判は、反証や例外を疑って粗を探す役。"
             "実務は、実現性やコストで、結局どれが使えるかを判断する役。"
             "探索は、仮説や比喩で別の見方を出して選択肢を増やす役です。"
             "実際にはこれらを短い日本語の指示文で与えているだけですが、"
             "大事なのは、同じ問題をあえて違う切り口で見る3体を用意した、という点です。"
             "このあとのデモで、この3つの個性が実際に効いている様子が見られます。")


def sl_shapley():
    s = add_slide(prs)
    title_head(s, "手法③", "貢献度の測り方：「その人が抜けたら、どれだけ困る？」")
    para_block(s, 0.85, 1.9, 6.6, 3.4, [
        [("チームを鍛えるには、"
          "個人の成績ではなく", 16, INK, FONT_REG, {}),
         ("チームへの貢献", 16, ACCENT, FONT_BOLD, {}),
         ("を測りたい。", 16, INK, FONT_REG, {})],
        [("そこで、", 15.5, INK, FONT_REG, {}),
         ("あらゆる組み合わせでチームを試し、"
          "「その1体が居るときと居ないときの差」を平均", 15.5, INK, FONT_BOLD, {}),
         ("します。", 15.5, INK, FONT_REG, {})],
        [("これは協力ゲーム理論の", 15, INK, FONT_REG, {}),
         ("シャープレイ値", 15, TEAL, FONT_BOLD, {}),
         ("という考え方。3体なら全7通りを実測でき、"
          "近似なしで公平に貢献度を配れます。", 15, INK, FONT_REG, {})],
    ], space_after=13)
    rect(s, 7.75, 1.95, 4.6, 3.55, fill=PRAG_BG, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.05)
    text(s, 8.05, 2.15, 4.0, 0.5, [("実際に起きたこと", 13, PRAG, FONT_BOLD, {})])
    para_block(s, 8.05, 2.65, 4.05, 2.8, [
        [("ある個体は、", 14, INK, FONT_REG, {}),
         ("単体の成績は3体で最低", 14, NEG, FONT_BOLD, {}),
         ("。", 14, INK, FONT_REG, {})],
        [("でも", 14, INK, FONT_REG, {}),
         ("チームに入れると成績が最も伸びた", 14, POS, FONT_BOLD, {}),
         ("。", 14, INK, FONT_REG, {})],
        [("→ 個人の点数だけ見ていたら"
          "捨てていた個体を、", 13.5, INK, FONT_REG, {}),
         ("貢献度は正しく拾い上げた", 13.5, TEAL, FONT_BOLD, {}),
         ("。", 13.5, INK, FONT_REG, {})],
    ], space_after=12, line_spacing=1.25)
    text(s, 0.85, 6.15, 11.6, 0.7, [("“優秀な個人の寄せ集め＝最強のチーム”とは限らない"
         "——アンサンブル学習の古典的な教訓とも一致します。", 13.5, SUB, FONT_REG, {})])
    foot(s)
    notes(s, "チームを鍛えるとき、個人の成績ではなくチームへの貢献を測りたいんです。"
             "そこで、あらゆる組み合わせでチームを試して、その1体が居るときと居ないときの差を平均します。"
             "協力ゲーム理論のシャープレイ値という考え方で、3体なら全7通りを実測できるので近似なしで公平に配れます。"
             "実際、面白いことが起きました。単体の成績は3体で最低なのに、チームに入れると一番伸びる個体があったんです。"
             "個人の点数だけ見ていたら捨てていた個体を、貢献度はちゃんと拾い上げた。"
             "優秀な個人の寄せ集めが最強とは限らない、という教訓そのものでした。")


def sl_evolution_method():
    s = add_slide(prs)
    title_head(s, "手法④", "進化：重みを“遺伝子”とみなして鍛える")
    para_block(s, 0.85, 1.95, 11.4, 1.35, [
        [("各AIの“性格”は、土台モデルに足す小さな重み（LoRA）で表せます。"
          "この重みを遺伝子とみなし、", 16, INK, FONT_REG, {}),
         ("貢献度の高い個体を選んで世代交代", 16, ACCENT, FONT_BOLD, {}),
         ("させます。", 16, INK, FONT_REG, {})],
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
        rect(s, x, 3.4, w, 2.05, fill=CARD, line=LINE, line_w=1.0,
             shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06, shadow=True)
        rect(s, x + 0.32, 3.65, 1.4, 0.5, fill=col, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.4)
        text(s, x + 0.32, 3.65, 1.4, 0.5, [(nm, 14, CARD, FONT_BOLD, {})],
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        text(s, x + 0.34, 4.3, w - 0.66, 1.0, [(body, 13, INK, FONT_REG, {})], line_spacing=1.25)
    text(s, 0.85, 5.95, 11.6, 0.8,
         [("これが研究の“看板”です。ただし——", 13.5, INK, FONT_REG, {}),
          ("実際に回してみたら予想と違う結果が出ました（後半で正直に報告します）。",
           13.5, ACCENT, FONT_BOLD, {})])
    foot(s)
    notes(s, "進化の中身です。各AIの性格は、土台モデルに足す小さな重み、LoRAで表せます。"
             "この重みを遺伝子とみなして、貢献度の高い個体を選んで世代交代させる。"
             "交叉は良い2体の重みを混ぜて子を作る。突然変異は少しゆらす。選択は貢献度の高い個体を残す。"
             "これが研究の看板です。ただし、実際に回してみたら予想と違う結果が出ました。"
             "そこは後半の結果のところで、正直に報告します。まずは手法の続きです。")


def sl_build_robust():
    s = add_slide(prs)
    title_head(s, "手法⑤", "落とし穴：性格を教えると“賢さ”が削れる")
    pic_fit(s, str(AST / "fig_cot_repair.png"), 0.85, 1.85, 7.2, 4.2, align="center", valign="top")
    para_block(s, 8.3, 2.05, 4.1, 4.0, [
        [("性格をむりに教え込むと、", 14.5, INK, FONT_REG, {}),
         ("考える途中の説明が半分に縮み", 14.5, NEG, FONT_BOLD, {}),
         ("、数学が解けなくなりました（−13点）。", 14.5, INK, FONT_REG, {})],
        [("原因は「短い答え方」を覚えすぎたこと。", 14, INK, FONT_REG, {})],
        [("対策 ＝ ", 14.5, INK, FONT_REG, {}),
         ("リプレイ（復習）", 14.5, TEAL, FONT_BOLD, {}),
         ("。元のモデル自身に“長い正しい解答”を"
          "作らせ、学習に混ぜて賢さを取り戻す。", 14.5, INK, FONT_REG, {})],
    ], space_after=13, line_spacing=1.28)
    foot(s)
    notes(s, "手法の5番目、ここは大事な落とし穴です。性格をむりに教え込むと副作用がありました。"
             "考える途中の説明が半分くらいに縮んでしまって、その分だけ数学が解けなくなった。マイナス13点です。"
             "原因は、短い答え方を覚えすぎたこと。"
             "対策がリプレイ、いわば復習です。元のモデル自身に長い正しい解答を作らせて、"
             "それを学習に混ぜる。性格を保ったまま、賢さを取り戻す。これで解けるようになりました。")


def sl_build_tricks():
    s = add_slide(prs)
    title_head(s, "手法⑥", "議論を“頑丈”にする3つの工夫")
    cards = [
        ("発言を匿名にする", CRITIC,
         "「誰の意見か」を隠して議論する。多数派や“偉い人”の意見に流されにくくする。"),
        ("むやみに変えさせない", PRAG,
         "自分の誤りを具体的に見つけられた時だけ、答えの変更を許す（条件つき更新）。"),
        ("自信を重みにする", EXPLORE,
         "多数決のとき、自信の高い答えを重く数える（重み付き投票）。"),
    ]
    y = 2.05
    for hd, col, body in cards:
        rect(s, 0.85, y, 11.5, 1.3, fill=CARD, line=LINE, line_w=1.0,
             shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.05, shadow=True)
        rect(s, 0.85, y, 0.14, 1.3, fill=col, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.5)
        text(s, 1.2, y + 0.18, 10.8, 0.5, [(hd, 16, col, FONT_BOLD, {})])
        text(s, 1.2, y + 0.7, 10.9, 0.5, [(body, 13.5, INK, FONT_REG, {})], line_spacing=1.15)
        y += 1.46
    text(s, 0.85, 6.5, 11.6, 0.5,
         [("狙いはどれも同じ——", 13.5, INK, FONT_REG, {}),
          ("“流されて正解を捨てる”のを防ぐ", 13.5, ACCENT, FONT_BOLD, {}),
          ("（このあとのデモで、崩れる実例が出ます）。", 13.5, INK, FONT_REG, {})])
    foot(s)
    notes(s, "手法の最後、議論を頑丈にする3つの工夫です。"
             "1つ目、発言を匿名にする。誰の意見かを隠して、多数派や偉い人の意見に流されにくくする。"
             "2つ目、むやみに変えさせない。自分の誤りを具体的に見つけられた時だけ、答えの変更を許す。"
             "3つ目、自信を重みにする。多数決のとき、自信の高い答えを重く数える。"
             "狙いはどれも同じで、流されて正解を捨てるのを防ぐこと。"
             "このあとのデモで、まさにその崩れる実例が出てきます。")


# ============================================================
# 章扉③ デモ
# ============================================================
def div_demo():
    s = section_divider(prs, 3, "実際の会話を見る",
                        "ここからは本物のログ。AIたちが議論して、正解にたどり着く（時に、崩れる）。")
    notes(s, "ここからが今日のメインです。実際にAIたちがどう会話しているか、本物のログをお見せします。"
             "うまくいく例、次に崩れる例。スライドを送るごとに会話が進みます。")


def sl_demo_intro():
    s = add_slide(prs)
    title_head(s, "デモの見方", "3体が同じ問題を解き、答えを見せ合って考え直す")
    steps = [
        ("① 各自が解く", "3体が別々に問題を解き、それぞれ答えを出す", INK_PANEL),
        ("② 見せ合う", "互いの答えと理由を読み、もう一度考え直す", TEAL),
        ("③ まとめる", "最終的な答えを重み付き投票で決める", POS),
    ]
    x0, w, gap = 0.85, 3.72, 0.19
    for i, (hd, body, col) in enumerate(steps):
        x = x0 + i * (w + gap)
        rect(s, x, 2.3, w, 2.0, fill=CARD, line=LINE, line_w=1.0,
             shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06, shadow=True)
        text(s, x + 0.32, 2.55, w - 0.6, 0.6, [(hd, 17, col, FONT_BOLD, {})])
        text(s, x + 0.34, 3.25, w - 0.66, 1.0, [(body, 13.5, INK, FONT_REG, {})], line_spacing=1.25)
        if i < 2:
            text(s, x + w - 0.02, 2.95, 0.28, 0.6, [("→", 18, SUB, FONT_BOLD, {})],
                 align=PP_ALIGN.CENTER)
    rect(s, 0.85, 4.7, 11.5, 1.5, fill=PAPER, line=ACCENT, line_w=1.3,
         shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.05)
    text(s, 1.2, 4.95, 10.9, 1.1, [
        ("これから見せるのは、", 14.5, INK, FONT_REG, {}),
        ("すべて本研究のLoRAチームが実際に記録した会話", 14.5, ACCENT, FONT_BOLD, {}),
        ("です（英語のやりとりを日本語に要約、原文の一部も併記）。\n"
         "最初は“うまくいく例”から。数学の問題です。", 14.5, INK, FONT_REG, {})],
        line_spacing=1.3)
    foot(s)
    notes(s, "デモの見方です。3ステップ。まず3体が別々に解いて答えを出す。"
             "次に互いの答えと理由を読んで考え直す。最後に重み付き投票で結論を決める。"
             "これから見せるのは、すべて本研究のLoRAチームが実際に記録した会話です。"
             "英語なので日本語に要約していますが、原文の一部も載せています。まずはうまくいく例、数学の問題から。")


def _cantor_question(s):
    rect(s, 0.85, 1.72, 11.5, 1.72, fill=CRITIC_BG, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.05)
    text(s, 1.2, 1.9, 11.0, 0.5, [("問題（数学）", 12.5, CRITIC, FONT_BOLD, {})])
    text(s, 1.2, 2.28, 11.0, 1.1, [
        ("「1/4 と 1/13 は “カントール集合” に入る？」", 18, INK, FONT_BOLD, {}),
        ("\nカントール集合 ＝ ざっくり言うと「3進数で書いたときに “1” が出てこない数」の集まり。", 14, SUB, FONT_REG, {}),
    ], line_spacing=1.25)
    rect(s, 9.9, 1.85, 2.3, 0.62, fill=POS, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.3)
    text(s, 9.9, 1.85, 2.3, 0.62, [("正解：両方入る", 13.5, CARD, FONT_BOLD, {})],
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)


def sl_cantor_q():
    s = add_slide(prs)
    title_head(s, "デモ・成功例 ①／④", "まず問題を確認する")
    _cantor_question(s)
    for i, (nm, col, bg) in enumerate([("批判", CRITIC, CRITIC_BG),
                                       ("実務", PRAG, PRAG_BG), ("探索", EXPLORE, EXPLORE_BG)]):
        x = 0.85 + i * 3.9
        rect(s, x, 4.0, 3.7, 2.15, fill=CARD, line=LINE, line_w=1.0,
             shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06, shadow=True)
        rect(s, x + 0.3, 4.25, 1.5, 0.44, fill=bg, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.4)
        text(s, x + 0.3, 4.25, 1.5, 0.44, [(nm, 12.5, col, FONT_BOLD, {})],
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        text(s, x, 5.0, 3.7, 0.9, [("考え中…", 15, FAINT, FONT_REG, {})],
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    foot(s)
    notes(s, "問題はこれです。4分の1と13分の1が、カントール集合に入るか。"
             "カントール集合とは、ざっくり言うと、3進数で書いたときに1が出てこない数の集まりです。"
             "正解は、実は両方とも入るんですが、これがなかなか難しい。3体がどう答えるか見てみましょう。")


def sl_cantor_r1():
    s = add_slide(prs)
    title_head(s, "デモ・成功例 ②／④", "① 各自が解く —— この時点では“不正解”が多数派")
    b = [
        ("批判", CRITIC, CRITIC_BG, "1/13 は3進数で1が出る…と判断", [
            [("1/4 は入る。でも 1/13 は3進数にすると "
              "“1” が出るはず → 入らない", 13, INK, FONT_REG, {})],
            [("“…contains a 1, so 1/13 does not belong.”", 10.5, FAINT, FONT_REG, {"italic": True})],
        ], "J", False),
        ("実務", PRAG, PRAG_BG, "展開して級数で検算 → 入る", [
            [("1/4 は入る。1/13 も 0.002002…(3進) と展開でき、"
              "足し合わせると確かに 1/13。だから入る", 13, INK, FONT_REG, {})],
            [("“= 2/26 = 1/13 … only 0s and 2s → belongs.”", 10.5, FAINT, FONT_REG, {"italic": True})],
        ], "H", True),
        ("探索", EXPLORE, EXPLORE_BG, "批判と同じく“1が出る”と判断", [
            [("1/13 の3進展開に “1” が見える → 入らない、と判断", 13, INK, FONT_REG, {})],
            [("“…contains a 1 … answers are [1, 0].”", 10.5, FAINT, FONT_REG, {"italic": True})],
        ], "J", False),
    ]
    y = 1.78
    for nm, col, bg, head, body, ans, ok in b:
        bubble(s, 0.85, y, 11.5, 1.42, nm, col, bg, head, body, answer=ans, answer_ok=ok)
        y += 1.56
    rect(s, 0.85, 6.5, 11.5, 0.5, fill=NEG, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.3)
    text(s, 0.85, 6.5, 11.5, 0.5, [
        ("いま多数決すると… 不正解の「J」が2票。正解は“実務”1体だけに埋もれている", 13, CARD, FONT_BOLD, {})],
        align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    foot(s)
    notes(s, "各自が解きました。批判と探索は、1/13を3進数にすると1が出るから入らない、と判断してJ。"
             "ところが実務だけは違った。1/13を0.00202…と展開して、足し合わせると本当に1/13になると検算までして、"
             "入る、と答えてH。これが正解です。"
             "でもこの時点で多数決すると、不正解のJが2票。正解が1体だけに埋もれている。あぶない状況です。"
             "普通の多数決だとここで負けます。ここから議論が始まります。")


def sl_cantor_r2():
    s = add_slide(prs)
    title_head(s, "デモ・成功例 ③／④", "② 見せ合う —— 検算を見た2体が“訂正”した")
    b = [
        ("批判", CRITIC, CRITIC_BG, "実務の検算を自分でも確かめた → 訂正", [
            [("2/27 + 2/729 + … を自分で計算したら確かに 1/13。"
              "“1” は出ない。答えを H に変える", 13, INK, FONT_REG, {})],
        ], "H", True),
        ("探索", EXPLORE, EXPLORE_BG, "批判の展開ミスに気づいた → 訂正", [
            [("さっきの3進展開は誤り。正しくは 0.002002… で “1” を含まない。"
              "H に変える", 13, INK, FONT_REG, {})],
        ], "H", True),
        ("実務", PRAG, PRAG_BG, "そのまま（正解を主張し続ける）", [
            [("展開と検算は合っている。両方カントール集合に入る＝H", 13, INK, FONT_REG, {})],
        ], "H", True),
    ]
    y = 2.0
    for nm, col, bg, head, body, ans, ok in b:
        bubble(s, 0.85, y, 11.5, 1.18, nm, col, bg, head, body, answer=ans, answer_ok=ok)
        y += 1.34
    rect(s, 0.85, 6.5, 11.5, 0.5, fill=POS, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.3)
    text(s, 0.85, 6.5, 11.5, 0.5, [
        ("正解の「H」が多数派に逆転 —— 少数派だった正解が、検算を通じてチームを動かした", 13, CARD, FONT_BOLD, {})],
        align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    foot(s)
    notes(s, "議論のラウンドです。批判は、実務の検算を自分でも確かめて、確かに1は出ない、と納得してHに訂正。"
             "探索も、さっきの3進展開が間違っていた、と気づいてHに訂正。実務はそのまま正解のHを主張。"
             "結果、正解のHが多数派に逆転しました。"
             "少数派だった正解が、検算という確かめられる根拠を通じてチーム全体を動かした。これが議論のいい面です。")


def sl_cantor_msg():
    s = add_slide(prs)
    title_head(s, "デモ・成功例 ④／④", "この例が示すこと")
    para_block(s, 0.85, 2.1, 11.4, 2.0, [
        [("多数決だけなら負けていた問題を、", 18, INK, FONT_REG, {}),
         ("議論がひっくり返して正解にした", 18, POS, FONT_BOLD, {}),
         ("。", 18, INK, FONT_REG, {})],
    ], space_after=10)
    rect(s, 0.85, 3.3, 11.5, 2.4, fill=PRAG_BG, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.05)
    text(s, 1.25, 3.6, 10.8, 2.0, [
        ("なぜうまくいったのか", 13.5, PRAG, FONT_BOLD, {}),
        ("\n数学は「答えを確かめられる」領域だから。", 17, INK, FONT_BOLD, {}),
        ("\n相手の主張が正しいかを自分で検算できるので、"
         "議論が“なんとなくの同調”ではなく“根拠にもとづく訂正”になる。", 15, INK, FONT_REG, {}),
        ("\n逆に言えば、確かめにくい問題では、この良さは出にくい——次はその例。", 14.5, SUB, FONT_REG, {}),
    ], line_spacing=1.3)
    foot(s)
    notes(s, "この例が示すのは、多数決だけなら負けていた問題を、議論がひっくり返して正解にした、ということです。"
             "なぜうまくいったか。数学は答えを確かめられる領域だからです。"
             "相手の主張が正しいかを自分で検算できるので、議論がなんとなくの同調ではなく、根拠にもとづく訂正になる。"
             "逆に言えば、確かめにくい問題では、この良さは出にくい。次はその、崩れてしまう例を見ます。")


def sl_marriage_q():
    s = add_slide(prs)
    title_head(s, "デモ・失敗例 ①／③", "次は“崩れる”例 —— 身近な常識問題")
    rect(s, 0.85, 1.72, 11.5, 1.5, fill=EXPLORE_BG, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.05)
    text(s, 1.2, 1.92, 11.0, 0.5, [("問題（常識・心理）", 12.5, EXPLORE, FONT_BOLD, {})])
    text(s, 1.2, 2.32, 11.0, 0.9, [
        ("「夫婦が長く連れ添える “いちばんの理由” は？」", 18, INK, FONT_BOLD, {}),
    ], line_spacing=1.2)
    rect(s, 9.7, 1.85, 2.5, 0.62, fill=POS, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.3)
    text(s, 9.7, 1.85, 2.5, 0.62, [("正解：親友だから", 13.5, CARD, FONT_BOLD, {})],
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    b = [
        ("批判", CRITIC, CRITIC_BG, "研究を引用しつつ“意見が合う”を選ぶ", [
            [("価値観の一致が長続きの予測因子（Gottman研究）→「ほぼ何でも意見が合う」", 13, INK, FONT_REG, {})]],
         "意見の一致", False),
        ("実務", PRAG, PRAG_BG, "感情のつながりを重視 → 親友", [
            [("長続きの土台は感情的な親密さ →「お互いが親友」", 13, INK, FONT_REG, {})]],
         "親友", True),
        ("探索", EXPLORE, EXPLORE_BG, "同じく親友を選ぶ", [
            [("深い結びつきの核は友情 →「お互いが親友」", 13, INK, FONT_REG, {})]],
         "親友", True),
    ]
    y = 3.5
    for nm, col, bg, head, body, ans, ok in b:
        bubble(s, 0.85, y, 11.5, 0.9, nm, col, bg, head, body, answer=ans, answer_ok=ok)
        y += 1.0
    rect(s, 0.85, 6.45, 11.5, 0.5, fill=POS, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.3)
    text(s, 0.85, 6.45, 11.5, 0.5, [("① 各自が解く：正解「親友」が2票 —— ここまでは正しい", 13, CARD, FONT_BOLD, {})],
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    foot(s)
    notes(s, "次は崩れる例です。問題は身近。夫婦が長く連れ添えるいちばんの理由は何か。"
             "正解は、お互いが親友だから、です。各自が解くと、実務と探索は親友と答えて正解。"
             "批判だけが、研究を引用して、価値観の一致＝意見が合うことだ、と答えて外した。"
             "この時点では正解の親友が2票。ここまでは正しいんです。問題は次のラウンドで起きます。")


def sl_marriage_r2():
    s = add_slide(prs)
    title_head(s, "デモ・失敗例 ②／③", "② 見せ合う —— “もっともらしい理屈”に流された")
    b = [
        ("実務", PRAG, PRAG_BG, "「親友は“一致”の結果では？」と考え直す → 転向", [
            [("“親友”は、価値観の一致から生まれる関係の結果。"
              "より根本的なのは「意見が合う」ことだ", 13, INK, FONT_REG, {})],
        ], "意見の一致", False),
        ("探索", EXPLORE, EXPLORE_BG, "同じ理屈に納得 → 転向", [
            [("友情は“合意”から生まれる状態にすぎない。"
              "土台は「意見が合う」ことのはず", 13, INK, FONT_REG, {})],
            [("“…best friends is a state that emerges from agreement.”",
              10.5, FAINT, FONT_REG, {"italic": True})],
        ], "意見の一致", False),
        ("批判", CRITIC, CRITIC_BG, "研究の引用で“意見の一致”を維持", [
            [("価値観の一致こそ予測因子（Gottman研究）＝「意見が合う」", 13, INK, FONT_REG, {})],
        ], "意見の一致", False),
    ]
    y = 2.35
    for nm, col, bg, head, body, ans, ok in b:
        h = 1.15 if nm == "探索" else 0.98
        bubble(s, 0.85, y, 11.5, h, nm, col, bg, head, body, answer=ans, answer_ok=ok)
        y += h + 0.2
    rect(s, 0.85, 6.5, 11.5, 0.5, fill=NEG, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.3)
    text(s, 0.85, 6.5, 11.5, 0.5, [
        ("正解「親友」が全滅（3体とも不正解へ）—— 正しかった多数派が理屈で崩れた", 13, CARD, FONT_BOLD, {})],
        align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    foot(s)
    notes(s, "議論のラウンドです。ここで悪い方向に転がります。"
             "実務が、いや“親友”というのは価値観の一致から生まれる結果であって、"
             "より根本的なのは意見が合うことだ、と考え直して転向。"
             "探索も、友情は合意から生まれる状態にすぎない、土台は意見が合うことだ、と同じ理屈に納得して転向。"
             "批判はもともと意見の一致派。結果、正解の親友が全滅して、3体とも不正解になってしまった。"
             "もっともらしい理屈、“より根本的な理由”という深読みに引きずられて、正しかった多数派が崩れたんです。")


def sl_demo_summary():
    s = add_slide(prs)
    title_head(s, "デモ・失敗例 ③／③", "2つのデモが示すこと")
    cols = [
        ("成功（数学）", POS, PRAG_BG,
         ["答えを確かめられる", "検算で“根拠ある訂正”", "少数派の正解が勝った"]),
        ("失敗（常識）", NEG, RGBColor(0xF6,0xEA,0xE6),
         ["答えを確かめにくい", "理屈で“同調”が起きる", "正しい多数派が崩れた"]),
    ]
    for i, (hd, col, bg, items) in enumerate(cols):
        x = 0.85 + i * 5.95
        rect(s, x, 2.15, 5.55, 2.9, fill=bg, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.05)
        rect(s, x, 2.15, 5.55, 0.66, fill=col, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.05)
        rect(s, x, 2.48, 5.55, 0.33, fill=col)
        text(s, x, 2.15, 5.55, 0.66, [(hd, 16, CARD, FONT_BOLD, {})],
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        bullets(s, x + 0.4, 3.05, 4.9, 1.9, items, size=14.5, gap=11,
                marker="●", marker_color=col)
    rect(s, 0.85, 5.35, 11.5, 1.35, fill=INK_PANEL, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.05)
    text(s, 1.25, 5.6, 10.8, 0.95, [
        ("同じ“議論”でも、結果は問題の性質で正反対になる。", 17, CARD, FONT_BOLD, {}),
        ("\nこの「領域による効き方の違い」こそ、実験でも一番はっきり出た発見でした。",
         14, RGBColor(0xC7,0xCE,0xD4), FONT_REG, {})],
        line_spacing=1.3)
    foot(s)
    notes(s, "2つのデモをまとめます。成功した数学の例は、答えを確かめられるので、検算による根拠ある訂正が起きて、"
             "少数派の正解が勝ちました。失敗した常識の例は、答えを確かめにくいので、理屈による同調が起きて、"
             "正しい多数派が崩れた。同じ議論でも、結果は問題の性質で正反対になるんです。"
             "この、領域による効き方の違いこそ、実験でも一番はっきり出た発見でした。ここから結果の話に移ります。")


# ============================================================
# 章扉④ 実験と結果
# ============================================================
def div_results():
    s = section_divider(prs, 4, "実験と結果",
                        "何を・どんな相手と・どんな環境で測ったか。そして、正直な結果。")
    notes(s, "ここからは実験と結果です。何を、どんな相手と、どんな環境で測ったか。そして正直な結果を報告します。")


def sl_setup():
    s = add_slide(prs)
    title_head(s, "実験設定", "何を・どんな相手と比べたか（4つの条件）")
    conds = [
        ("① ベース単体", BASE_D, "土台モデル1体が、普通に1回解く。", "土台の実力（出発点）"),
        ("② 素の議論", SUB, "性格なしの同じモデル3体で議論。", "議論そのものの効果を見る"),
        ("③ 多数決9回（SC@9）", SC9_D, "同じモデルが9回バラバラに解いて多数決。", "最強の比較相手"),
        ("④ LoRAチーム（本命）", TEAL, "3つの性格を持たせ、賢さを保って議論・投票するチーム。", "本研究の提案"),
    ]
    x0, y0, w, hh, gx, gy = 0.85, 1.95, 5.62, 1.5, 0.28, 0.22
    for i, (hd, col, body, tag) in enumerate(conds):
        x = x0 + (i % 2) * (w + gx)
        y = y0 + (i // 2) * (hh + gy)
        rect(s, x, y, w, hh, fill=CARD, line=LINE, line_w=1.0,
             shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06, shadow=True)
        rect(s, x, y, 0.14, hh, fill=col, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.5)
        text(s, x + 0.32, y + 0.14, w - 0.5, 0.4, [(hd, 15, col, FONT_BOLD, {})])
        text(s, x + 0.34, y + 0.58, w - 0.6, 0.5, [(body, 12.5, INK, FONT_REG, {})], line_spacing=1.12)
        text(s, x + 0.34, y + 1.08, w - 0.6, 0.35, [("→ " + tag, 11.5, col, FONT_MED, {})])
    rect(s, 0.85, 5.55, 11.5, 1.15, fill=PAPER, line=LINE, line_w=1.0,
         shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.05)
    text(s, 1.15, 5.72, 11.0, 0.9, [
        ("3つのテスト：", 12.5, INK, FONT_BOLD, {}),
        ("一般知識・推論（MMLU-Pro）／ 数学（MATH-500）／ 大学院級の科学（SuperGPQA）。　", 12.5, INK, FONT_REG, {}),
        ("計算環境：", 12.5, INK, FONT_BOLD, {}),
        ("クラウド上のGPU。\n公平のため、同じ問題で条件どうしを“対戦”させ、"
         "統計検定（多重比較の補正つき）で判定。乱数の種を変えても結果が変わらないかも確認。", 12.5, INK, FONT_REG, {}),
    ], line_spacing=1.3)
    foot(s)
    notes(s, "実験設定です。比べた条件は4つ。"
             "①ベース単体は、土台モデル1体が普通に1回解く、出発点。"
             "②素の議論は、性格なしの同じモデル3体で議論して、議論そのものの効果を見る。"
             "③多数決9回は、同じモデルが9回バラバラに解いて多数決する、最強の比較相手。"
             "④LoRAチームが本命で、3つの性格を持たせ、賢さを保って議論・投票するチームです。"
             "テストは3種類、一般知識・数学・大学院級の科学。計算はクラウドのGPU。"
             "公平のため、同じ問題で条件どうしを対戦させ、多重比較の補正つきで検定しています。")


def sl_finding_domain():
    s = add_slide(prs)
    title_head(s, "発見①", "議論が効く問題・効かない問題がある（領域依存）")
    pic_fit(s, str(AST / "fig_domain_flip.png"), 0.85, 1.75, 6.7, 4.6, align="center")
    para_block(s, 7.8, 2.1, 4.6, 4.0, [
        [("同じ「素の議論」でも、", 15, INK, FONT_REG, {}),
         ("問題の種類で結果が逆転", 15, ACCENT, FONT_BOLD, {}),
         ("しました。", 15, INK, FONT_REG, {})],
        [("数学や科学では議論すると上がる。"
          "でも一般知識では、むしろ下がってしまう。", 14.5, INK, FONT_REG, {})],
        [("鍵は", 14.5, INK, FONT_REG, {}),
         ("「答えを確かめられるか」", 14.5, TEAL, FONT_BOLD, {}),
         ("。確かめられる問題では、議論が正しい方向に働く"
          "（さっきの数学デモと同じ）。", 14.5, INK, FONT_REG, {})],
        [("この“反転”は、先行研究の食い違い"
          "（議論は効く／効かない）も説明できます。", 13.5, SUB, FONT_REG, {})],
    ], space_after=13, line_spacing=1.25)
    foot(s)
    notes(s, "1つ目の発見です。同じ素の議論でも、問題の種類で結果が逆転しました。"
             "数学や科学では議論すると上がる。でも一般知識では、むしろ下がってしまう。"
             "鍵は、答えを確かめられるかどうか。確かめられる問題では議論が正しい方向に働く。さっきの数学デモと同じ理屈です。"
             "この反転は、議論は効くという研究と効かないという研究の食い違いも、うまく説明できます。")


def sl_finding_evolution():
    s = add_slide(prs)
    title_head(s, "発見②（看板の答え合わせ）", "進化は“強化”ではなく“修復”だった")
    pic_fit(s, str(AST / "fig_evolution.png"), 0.85, 1.8, 7.0, 4.4, align="center", valign="top")
    para_block(s, 8.1, 2.05, 4.3, 4.2, [
        [("看板だった進化を6世代回した結果——", 14.5, INK, FONT_REG, {})],
        [("動いたのは数学だけ（+4.3点）。"
          "他はほぼゼロ。", 14.5, NEG, FONT_BOLD, {})],
        [("しかも重みの“向き”だけが回り、"
          "“大きさ”は変わっていなかった。", 14, INK, FONT_REG, {})],
        [("＝進化は個体を強くしたのではなく、"
          "性格を教えた時の“壊れ”を少し戻す", 14, INK, FONT_REG, {}),
         ("修復として働いた", 14, TEAL, FONT_BOLD, {}),
         ("。", 14, INK, FONT_REG, {})],
        [("→ 同じ修復は、リプレイ（復習）の方が"
          "確実。だから最良チームはリプレイで仕上げた。", 13.5, SUB, FONT_REG, {})],
    ], space_after=11, line_spacing=1.25)
    foot(s)
    notes(s, "2つ目、これは研究の看板の答え合わせです。正直に言います。"
             "看板だった進化を6世代回した結果、動いたのは数学だけ、プラス4.3点。他はほぼゼロでした。"
             "しかも中を調べると、重みの向きだけが回って、大きさは変わっていなかった。"
             "つまり進化は、個体を強くしたのではなく、性格を教えた時に壊れた部分を少し戻す、修復として働いていたんです。"
             "期待した選抜して強くする、ではなかった。"
             "そして同じ修復は、さっき説明したリプレイ、復習の方が確実にできた。"
             "だから最終的な最良チームは、進化ではなくリプレイで仕上げています。ここは負けを含む、正直な報告です。")


def sl_final():
    s = add_slide(prs)
    title_head(s, "最終結果", "LoRAチームは“ベースと互角”。ただし多数決9回には届かない")
    pic_fit(s, str(AST / "fig_final_results.png"), 0.85, 1.7, 7.9, 4.5, align="center", valign="top")
    para_block(s, 8.7, 1.95, 3.75, 2.6, [
        [("LoRAチーム（青）は3種目とも", 13, INK, FONT_REG, {}),
         ("ベース単体と互角", 13, POS, FONT_BOLD, {}),
         ("。", 13, INK, FONT_REG, {})],
        [("数学では", 13, INK, FONT_REG, {}),
         ("多数決9回に肩を並べた", 13, TEAL, FONT_BOLD, {}),
         ("。", 13, INK, FONT_REG, {})],
    ], space_after=10, line_spacing=1.25)
    rect(s, 8.7, 4.15, 3.75, 2.05, fill=PAPER, line=LINE, line_w=1.1,
         shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06)
    para_block(s, 8.95, 4.32, 3.3, 1.8, [
        [("総合の判定", 12, ACCENT, FONT_BOLD, {})],
        [("○ ベース単体と互角", 13, POS, FONT_BOLD, {})],
        [("× 多数決9回に −3.0点", 13, NEG, FONT_BOLD, {})],
        [("＝工夫を尽くしても素朴な強敵に勝てなかった", 11.5, SUB, FONT_REG, {})],
    ], space_after=8, line_spacing=1.2)
    foot(s)
    notes(s, "最終結果です。測定環境をそろえて、3条件を比べました。"
             "LoRAチーム、青い棒は、3種目ともベース単体と互角。特に数学では、最強の多数決9回に肩を並べています。"
             "ただし総合では、多数決9回にマイナス3.0点で届きませんでした。"
             "工夫を尽くしても、素朴な強敵には勝てなかった。これが正直な結果です。"
             "でも次のスライドで話しますが、これはダメだった、で終わる話ではありません。")


# ============================================================
# 章扉⑤ 考察・結論・今後
# ============================================================
def div_wrap():
    s = section_divider(prs, 5, "考察・結論・今後",
                        "なぜこうなったのか。何が言えて、次に何をするか。")
    notes(s, "最後に、考察と結論、今後です。なぜこうなったのか、何が言えて、次に何をするかを話します。")


def sl_discuss_why():
    s = add_slide(prs)
    title_head(s, "考察①", "なぜ勝てないのか？ でも、どこに価値があるのか？")
    pic_fit(s, str(AST / "fig_budget.png"), 0.85, 1.75, 7.2, 3.6, align="center", valign="top")
    para_block(s, 8.2, 2.0, 4.2, 3.6, [
        [("多数決9回は、", 14.5, INK, FONT_REG, {}),
         ("独立した9票", 14.5, SUB, FONT_BOLD, {}),
         ("の“数の力”が強い。", 14.5, INK, FONT_REG, {})],
        [("チームは実質3つの視点しかなく、"
          "議論でその差を埋めきれなかった。", 14, INK, FONT_REG, {})],
        [("ただし", 14.5, INK, FONT_REG, {}),
         ("数学では互角", 14.5, POS, FONT_BOLD, {}),
         ("。確かめられる問題では、少ない視点でも議論が補える。", 14, INK, FONT_REG, {})],
    ], space_after=14, line_spacing=1.28)
    rect(s, 0.85, 5.7, 11.5, 1.05, fill=PRAG_BG, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.05)
    text(s, 1.2, 5.88, 10.9, 0.7, [
        ("要するに——", 14.5, PRAG, FONT_BOLD, {}),
        ("“数で押す”のが有利な問題と、“話し合い”が活きる問題は別。使い分けが本質。",
         14.5, INK, FONT_REG, {})], line_spacing=1.2)
    foot(s)
    notes(s, "なぜ勝てないのか。多数決9回は、独立した9票の数の力が強いんです。"
             "一方チームは、3体×2ラウンドとはいえ、実質3つの視点しかない。議論でその差を埋めきれなかった。"
             "ただし数学では互角でした。確かめられる問題では、視点が少なくても議論が補えるんです。"
             "要するに、数で押すのが有利な問題と、話し合いが活きる問題は別で、使い分けが本質だ、ということです。")


def sl_discuss_lesson():
    s = add_slide(prs)
    title_head(s, "考察②", "研究を通じて分かった、2つの大事なこと")
    cards = [
        ("看板の進化は“修復”にとどまった", TEAL,
         "「貢献度で選抜して強くする」という当初の狙いは出なかった。進化は壊れた能力を少し戻す働き"
         "で、同じことはリプレイ（復習）の方が確実だった。看板の限界を、実データで突き止めた。"),
        ("同じ設定でも“測る環境”で点数が動く", ACCENT,
         "評価プログラムを入れ替えただけで、同じ問題・同じ設定なのに6点ほどズレる現象を発見。"
         "しかもベース側だけ過小評価されやすい。→ 比較は必ず同じ環境で。方法論としての教訓。"),
    ]
    y = 2.05
    for hd, col, body in cards:
        rect(s, 0.85, y, 11.5, 2.05, fill=CARD, line=LINE, line_w=1.0,
             shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.05, shadow=True)
        rect(s, 0.85, y, 0.14, 2.05, fill=col, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.5)
        text(s, 1.2, y + 0.2, 11.0, 0.5, [(hd, 16.5, col, FONT_BOLD, {})])
        text(s, 1.2, y + 0.78, 10.9, 1.1, [(body, 14, INK, FONT_REG, {})], line_spacing=1.3)
        y += 2.25
    foot(s)
    notes(s, "考察の2つ目。研究を通じて分かった、大事なことが2つあります。"
             "1つ目。看板だった進化は、修復にとどまりました。貢献度で選抜して強くするという当初の狙いは出ず、"
             "壊れた能力を少し戻す働きで、同じことはリプレイの方が確実だった。看板の限界を実データで突き止めた、とも言えます。"
             "2つ目。同じ問題・同じ設定なのに、評価プログラムを入れ替えただけで6点ほどズレる現象を見つけました。"
             "しかもベース側だけ過小評価されやすい。比較は必ず同じ環境で、というのは方法論としての教訓です。"
             "危うく間違った結論を出すところでした。")


def sl_conclusion():
    s = add_slide(prs)
    title_head(s, "結論", "この研究で分かったこと")
    items = [
        ("チームレベルの貢献度で鍛える枠組みを作り、実際に動かした",
         "3体なら貢献度を近似なしで公平に測れる。仕組みとして初めて形にした。"),
        ("議論の効き方は「領域」で決まる",
         "確かめられる問題では効き、そうでない問題では逆効果。統一的に説明できた。"),
        ("看板の進化は“修復”だと突き止め、賢さは別の方法で取り戻した",
         "進化は強化でなく修復。能力の毀損はリプレイ（復習）でベース水準まで回復させた。"),
        ("それでも“多数決9回”には勝てない、を厳密に確認（正直な負け）",
         "条件を公平にし、環境の罠も統制した上での結論。数学だけは互角。"),
    ]
    y = 1.95
    for i, (hd, body) in enumerate(items):
        rect(s, 0.85, y, 11.5, 1.12, fill=CARD, line=LINE, line_w=1.0,
             shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.05, shadow=True)
        rect(s, 1.05, y + 0.28, 0.56, 0.56, fill=TEAL, shape=MSO_SHAPE.OVAL)
        text(s, 1.05, y + 0.28, 0.56, 0.56, [(str(i+1), 17, CARD, FONT_BOLD, {})],
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        text(s, 1.85, y + 0.16, 10.3, 0.5, [(hd, 15, INK, FONT_BOLD, {})])
        text(s, 1.85, y + 0.62, 10.3, 0.45, [(body, 12.5, SUB, FONT_REG, {})])
        y += 1.24
    foot(s)
    notes(s, "結論です。4点にまとめました。"
             "1つ目。チームへの貢献度で鍛える枠組みを作って、実際に動かしました。3体なら近似なしで公平に測れる、を初めて形にした。"
             "2つ目。議論の効き方は領域で決まる。確かめられる問題では効いて、そうでないと逆効果、と統一的に説明できました。"
             "3つ目。看板の進化は修復だと突き止め、賢さはリプレイという別の方法で取り戻した。"
             "4つ目。それでも多数決9回には勝てない、を厳密に確認した。正直な負けですが、"
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
        text(s, 1.2, y + 0.2, 11.0, 0.5, [("▶ " + hd, 16, INK, FONT_BOLD, {})])
        text(s, 1.5, y + 0.72, 10.6, 0.5, [(body, 13.5, SUB, FONT_REG, {})], line_spacing=1.2)
        y += 1.5
    foot(s)
    notes(s, "今後の展開です。3つ。"
             "1つ目、いいとこ取りを試す。各メンバーが数回ずつ解いてから議論する、数の力と話し合いの合わせ技。"
             "今回の結果が指す、最有力の次の一手だと思っています。"
             "2つ目、進化のやり方を鍛え直す。選抜のノイズを抑えた改良版で、進化が修復を超えて本当の発見になるか。"
             "3つ目、違う性格や違うモデルを混ぜる。本当に異質なメンバーを組ませて、多様性を最大化したい。")


def sl_closing():
    s = add_slide(prs, bg=INK_PANEL)
    rect(s, 0, 0, 0.28, 7.5, fill=ACCENT)
    text(s, 1.2, 2.15, 11.0, 1.0, [("まとめ", 15, ACCENT, FONT_BOLD, {})])
    para_block(s, 1.2, 2.7, 10.8, 3.0, [
        [("小さなAIを3体、話し合わせて鍛えた。", 26, CARD, FONT_BLACK, {})],
        [("議論には“効く問題”と“効かない問題”があり、", 19, RGBColor(0xE7,0xEC,0xEF), FONT_REG, {})],
        [("看板の進化は“修復”と分かり、賢さは復習で取り戻した。", 19, RGBColor(0xE7,0xEC,0xEF), FONT_REG, {})],
        [("——ただし、素朴な強敵に勝つのはこれからの宿題。", 18, RGBColor(0xF0,0xD9,0xC7), FONT_MED, {})],
    ], space_after=14, line_spacing=1.2)
    line(s, 1.25, 6.1, 4.0, 0, color=ACCENT, weight=1.5)
    text(s, 1.2, 6.3, 11.0, 0.5, [("ご清聴ありがとうございました。", 15, CARD, FONT_REG, {})])
    notes(s, "まとめます。小さなAIを3体、話し合わせて鍛えました。"
             "議論には効く問題と効かない問題があって、看板の進化は修復だと分かり、賢さは復習で取り戻した。"
             "ただし、素朴な強敵に勝つのは、これからの宿題です。"
             "以上で発表を終わります。ご清聴ありがとうございました。")


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
        ("LoRAチーム（本命）", "71.6", "86.7", "43.1", "ベースと互角・数学は差なし"),
    ]
    x0, y0 = 0.95, 2.15
    ws = [3.4, 1.8, 1.6, 1.6, 3.4]
    rh = 0.66
    for r, row in enumerate(rows):
        x = x0
        for c, cell in enumerate(row):
            head = (r == 0)
            fill = INK_PANEL if head else (CARD if r % 2 else PAPER)
            rect(s, x, y0 + r * rh, ws[c], rh, fill=fill, line=LINE, line_w=0.8)
            tcol = CARD if head else INK
            fnt = FONT_BOLD if (head or c == 0) else FONT_REG
            emph = (r == 3)
            if emph and not head:
                tcol = TEAL if c > 0 else INK
                fnt = FONT_BOLD
            text(s, x + 0.14, y0 + r * rh, ws[c] - 0.24, rh,
                 [(cell, 12.5 if head else 12, tcol, fnt, {})],
                 align=PP_ALIGN.LEFT if c in (0, 4) else PP_ALIGN.CENTER,
                 anchor=MSO_ANCHOR.MIDDLE)
            x += ws[c]
    text(s, 0.95, 5.35, 11.4, 1.3, [
        ("・数学（MATH-500）では差は検出されず（−0.8点, 有意差なし。±2点の同等性は未確定）。\n", 12.5, INK, FONT_REG, {}),
        ("・総合（6,000問）では、LoRAチームは多数決9回に −3.0点（有意）。ベース単体とは互角（+0.3点）。\n", 12.5, INK, FONT_REG, {}),
        ("・多数決9回とLoRAチームは6シード、ベース単体・素の議論などは3シードで測定。", 12, SUB, FONT_REG, {}),
    ], line_spacing=1.3)
    apx_foot(s, 1)
    notes(s, "付録の数値表です。質疑用です。測定環境をそろえた最終値で、"
             "LoRAチームはベースと互角、数学では多数決9回とも互角です。ただ総合では多数決9回に3点届かない。ここが正直なところです。")


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
        text(s, 1.1, y + 0.34, 1.5, 0.48, [(nm, 14, CARD, FONT_BOLD, {})],
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        text(s, 2.9, y + 0.2, 9.2, 0.8, [("「" + txt + "」", 15, INK, FONT_REG, {})],
             anchor=MSO_ANCHOR.MIDDLE, line_spacing=1.2)
        y += 1.33
    text(s, 0.85, 6.35, 11.6, 0.5,
         [("たった1文ずつ。これだけで、デモで見たような異なる解き方が生まれます。", 13.5, SUB, FONT_REG, {})])
    apx_foot(s, 2)
    notes(s, "3つの性格に与えた実際の指示文です。それぞれたった1文ずつ。"
             "この短い指示だけで、デモで見たような異なる解き方が生まれる、というのは面白い点だと思います。")


def sl_apx_terms():
    s = add_slide(prs)
    title_head(s, "付録 A3", "用語ミニ辞典")
    terms = [
        ("LoRA", "土台モデルに足す小さな“追加重み”。少ない容量で性格や技能を足せる。"),
        ("リプレイ（復習）", "元モデル自身の長い正しい解答を学習に混ぜ、賢さの毀損を防ぐ手法。"),
        ("SC@9 / 多数決9回", "同じAIに9回バラバラに解かせ、多数決を取る素朴で強い方法。"),
        ("Shapley値（貢献度）", "「その人が抜けたらどれだけ困るか」を全組み合わせで公平に測った値。"),
        ("MMLU-Pro / MATH-500 / SuperGPQA", "順に、一般知識・推論／数学／大学院級科学の標準テスト。"),
    ]
    y = 2.0
    for t, d in terms:
        rect(s, 0.85, y, 3.7, 0.82, fill=INK_PANEL, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08)
        text(s, 1.05, y, 3.3, 0.82, [(t, 12.5, CARD, FONT_BOLD, {})],
             anchor=MSO_ANCHOR.MIDDLE, line_spacing=1.05)
        rect(s, 4.7, y, 7.65, 0.82, fill=CARD, line=LINE, line_w=0.9,
             shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08)
        text(s, 4.95, y, 7.2, 0.82, [(d, 13, INK, FONT_REG, {})],
             anchor=MSO_ANCHOR.MIDDLE, line_spacing=1.15)
        y += 0.95
    apx_foot(s, 3)
    notes(s, "最後に用語のミニ辞典です。LoRA、リプレイ、多数決9回、シャープレイ値、各テスト。"
             "本編で噛み砕いた言葉の、正式な対応関係をまとめています。質疑の参考にどうぞ。")


def build_all():
    cover(); sl_intro()
    div_background(); sl_bg_scale(); sl_bg_debate(); sl_bg_critique(); sl_bg_gap()
    div_method(); sl_method_overview(); sl_personas(); sl_shapley()
    sl_evolution_method(); sl_build_robust(); sl_build_tricks()
    div_demo(); sl_demo_intro()
    sl_cantor_q(); sl_cantor_r1(); sl_cantor_r2(); sl_cantor_msg()
    sl_marriage_q(); sl_marriage_r2(); sl_demo_summary()
    div_results(); sl_setup(); sl_finding_domain(); sl_finding_evolution(); sl_final()
    div_wrap(); sl_discuss_why(); sl_discuss_lesson(); sl_conclusion(); sl_future(); sl_closing()
    div_appendix(); sl_apx_numbers(); sl_apx_persona(); sl_apx_terms()


if __name__ == "__main__":
    build_all()
    out = HERE / "研究進捗_20260709_v2.pptx"
    prs.save(str(out))
    print("saved", out, "slides:", len(prs.slides._sldIdLst), "pages:", _page["n"])
