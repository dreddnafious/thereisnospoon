"""
Build circuits-primer.pdf from circuits-primer.md, as a typeset book rather than a printed web page.

Structure: cover -> contents (with real page numbers) -> orientation -> per part: an opener page, then one page-start
per numbered section, closing with a Holds/Qualifiers card -> back matter (key insights, references, figure notes).

Markdown -> structured HTML (python-markdown) + print CSS -> PDF via headless Chrome. Two passes: the first finds
the page each section lands on (pdftotext), the second prints those numbers into the contents and part openers.
Requires: `pip install markdown`; google-chrome or chromium; pdftotext (poppler-utils) for page numbers.
Fonts: IBM Plex Serif / Sans / Mono (falls back to system serif/sans if absent).
Run from the repo root:  python3 scripts/build_circuits_pdf.py
"""
import html
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata

import markdown

SRC, OUT_PDF = 'circuits-primer.md', 'circuits-primer.pdf'
REPO = 'https://github.com/dreddnafious/thereisnospoon/blob/main/'
WORDS = ['One', 'Two', 'Three', 'Four', 'Five', 'Six']
BACK_MATTER = {'Key Insights', 'References', 'Figures'}

# ---------------------------------------------------------------- parsing


def slug(value):
    value = unicodedata.normalize('NFKC', value).lower()
    value = ''.join(ch for ch in value if ch.isalnum() or ch in ' -')
    return re.sub(r'\s+', '-', value.strip())


def strip_icon(title):
    """Drop the leading emoji the GitHub version uses as a section icon."""
    return re.sub(r'^[^\w"(]+', '', title).strip()


def loosen_lists(text):
    """GitHub renders a list right under a paragraph line; python-markdown needs a blank line first."""
    out, in_code = [], False
    item = re.compile(r'^\s*([-*]|\d+\.)\s')
    for line in text.split('\n'):
        line = re.sub(r'^ {2,3}([-*] )', r'    \1', line)
        if line.startswith('```'):
            in_code = not in_code
        prev = out[-1] if out else ''
        if (not in_code and item.match(line) and prev.strip() and not item.match(prev)
                and not prev.startswith((' ', '|', '>'))):
            out.append('')
        out.append(line)
    return '\n'.join(out)


def clean(text):
    text = text.replace('](ml-primer.md', '](' + REPO + 'ml-primer.md').replace('](scripts/', '](' + REPO + 'scripts/')
    lines = [l for l in text.split('\n') if l.strip() not in ('<br>', '---')]
    return '\n'.join(lines)


def parse(md_text):
    md_text = clean(md_text)
    head, *part_chunks = re.split(r'^# (?=Part \d)', md_text, flags=re.M)
    # preamble: drop the title and "A companion to" lines (the cover carries them)
    pre = re.sub(r'^# Circuits and Formation.*\n', '', head, count=1, flags=re.M)
    pre = re.sub(r'^\*A companion to.*\*\s*\n', '', pre, count=1, flags=re.M)
    parts, back = [], []
    for chunk in part_chunks:
        title_line, body = chunk.split('\n', 1)
        num = int(re.match(r'Part (\d+)', title_line).group(1))
        ptitle = re.sub(r'^Part \d+\s*[—-]\s*', '', title_line).strip()
        lead_m = re.match(r'\s*\*(.+?)\*\s*\n', body)
        lead = lead_m.group(1) if lead_m else ''
        body = body[lead_m.end():] if lead_m else body
        holds = ''
        hm = re.search(r'^### Holds / Qualifiers.*$', body, flags=re.M)
        if hm:
            holds, body = body[hm.end():], body[:hm.start()]
        sections = []
        for sec in re.split(r'^## ', body, flags=re.M)[1:]:
            stitle, sbody = sec.split('\n', 1)
            stitle = strip_icon(stitle)
            (back if stitle in BACK_MATTER else sections).append({'title': stitle, 'body': sbody})
        # anything after a back-matter heading inside this part stays back matter
        parts.append({'num': num, 'title': ptitle, 'lead': lead, 'sections': sections, 'holds': holds})
    return pre, parts, back

# ---------------------------------------------------------------- rendering


def md_to_html(text):
    text = re.sub(r'^(#{3,} )[^\w\s]+\s*', r'\1', text, flags=re.M)   # drop heading emoji
    return markdown.markdown(loosen_lists(text), extensions=['tables', 'fenced_code', 'sane_lists'])


def decorate(body_html, part_num, fig_counter):
    """Figures with numbered captions; callouts; internal links."""
    def fig(m):
        alt, src = html.unescape(m.group(1)), m.group(2)
        fig_counter[0] += 1
        label = f'Figure {part_num}.{fig_counter[0]}'
        return (f'<figure class="fig"><img src="{src}" alt=""><figcaption><span class="fl">{label}</span>'
                f'{html.escape(alt)}</figcaption></figure>')
    body_html = re.sub(r'<p><img alt="([^"]*)" src="([^"]+)"\s*/?></p>', fig, body_html)
    body_html = re.sub(r'<blockquote>\s*<p><strong>Key insight:</strong>\s*',
                       '<aside class="insight"><div class="lbl">Key insight</div><p>', body_html)
    body_html = re.sub(r'<blockquote>\s*<p><strong>How to read this\.</strong>\s*',
                       '<aside class="note"><div class="lbl">How to read this</div><p>', body_html)
    body_html = re.sub(r'(<aside class="(?:insight|note)">.*?)</blockquote>', r'\1</aside>', body_html, flags=re.S)
    body_html = re.sub(r'href="#-([^"]+)"', r'href="#sec-\1"', body_html)
    return body_html


def holds_card(holds_md):
    holds, quals = [], []
    for item in re.findall(r'^- (.+?)(?=^- |\Z)', holds_md.strip() + '\n', flags=re.M | re.S):
        item = ' '.join(item.split())
        if item.startswith('**Holds:**'):
            holds.append(item[len('**Holds:**'):].strip())
        elif item.startswith('**Qualifier:**'):
            quals.append(item[len('**Qualifier:**'):].strip())
    li = lambda xs: ''.join(f'<li>{markdown.markdown(x)[3:-4]}</li>' for x in xs)
    return ('<div class="holds"><div class="col ok"><div class="lbl">What holds</div><ul>' + li(holds) +
            '</ul></div><div class="col q"><div class="lbl">Qualifiers</div><ul>' + li(quals) + '</ul></div></div>')


COVER_ART = """
<svg class="art" viewBox="0 0 600 150" xmlns="http://www.w3.org/2000/svg">
  <line x1="20" y1="75" x2="580" y2="75" stroke="#1f5f8b" stroke-width="6" stroke-linecap="round"/>
  <g stroke-width="2.4" fill="none">
    <line x1="80" y1="128" x2="80" y2="84" stroke="#1f5f8b"/>
    <line x1="170" y1="84" x2="170" y2="128" stroke="#8a94a0" stroke-dasharray="5 4"/>
    <line x1="195" y1="128" x2="195" y2="84" stroke="#2f8a5b"/>
    <line x1="285" y1="84" x2="285" y2="128" stroke="#8a94a0" stroke-dasharray="5 4"/>
    <line x1="310" y1="128" x2="310" y2="84" stroke="#7a4fa0"/>
    <line x1="400" y1="84" x2="400" y2="128" stroke="#8a94a0" stroke-dasharray="5 4"/>
    <line x1="425" y1="128" x2="425" y2="84" stroke="#2f8a5b"/>
    <line x1="530" y1="84" x2="530" y2="128" stroke="#8a94a0" stroke-dasharray="5 4"/>
    <path d="M 90 64 C 200 6, 330 6, 420 64" stroke="#c8922a" stroke-width="2.6"/>
  </g>
  <g fill="#fbfaf6" stroke-width="2.4">
    <circle cx="80" cy="134" r="7" stroke="#1f5f8b"/><circle cx="182" cy="134" r="7" stroke="#2f8a5b"/>
    <circle cx="297" cy="134" r="7" stroke="#7a4fa0"/><circle cx="412" cy="134" r="7" stroke="#2f8a5b"/>
    <circle cx="530" cy="134" r="7" stroke="#b03a2e"/>
  </g>
</svg>"""


def build_html(pre, parts, back, pages):
    pg = lambda key: pages.get(key, '')
    out = []
    # cover
    out.append(f"""
<section class="cover">
  <div class="kicker">A companion to <em>There Is No Spoon</em> · a machine learning primer</div>
  <h1 class="title">Circuits<br>and Formation</h1>
  <div class="subtitle">How networks grow their functions</div>
  {COVER_ART}
  <ol class="cover-parts">{''.join(f'<li><span>{p["num"]:02d}</span>{html.escape(p["title"])}</li>' for p in parts)}</ol>
  <div class="foot">github.com/dreddnafious/thereisnospoon</div>
</section>""")
    # contents
    rows = []
    for p in parts:
        rows.append(f'<div class="toc-part"><span class="n">{p["num"]:02d}</span><span class="t">{html.escape(p["title"])}'
                    f'</span><span class="pg">{pg(f"P{p["num"]}")}</span></div>')
        for i, s in enumerate(p['sections'], 1):
            rows.append(f'<div class="toc-sec"><span class="n">{p["num"]}.{i}</span><span class="t">{html.escape(s["title"])}'
                        f'</span><span class="dots"></span><span class="pg">{pg(f"S{p["num"]}.{i}")}</span></div>')
    rows.append(f'<div class="toc-part back"><span class="n">··</span><span class="t">Key insights · References'
                f'</span><span class="pg">{pg("BACK")}</span></div>')
    out.append(f'<section class="toc"><div class="eyebrow">Contents</div>{"".join(rows)}</section>')
    # orientation
    out.append(f'<section class="sec front"><span class="mk">@@ABOUT@@</span><div class="sec-head"><div class="sn">Before you start</div>'
               f'<h2>About this companion</h2></div><div class="body">{decorate(md_to_html(pre), 0, [0])}</div></section>')
    # parts
    for p in parts:
        n, figs = p['num'], [0]
        sec_list = ''.join(f'<li><span class="n">{n}.{i}</span><span class="t">{html.escape(s["title"])}</span>'
                           f'<span class="pg">{pg(f"S{n}.{i}")}</span></li>' for i, s in enumerate(p['sections'], 1))
        out.append(f"""
<div class="part p{n}">
<section class="opener"><span class="mk">@@P{n}@@</span>
  <div class="big">{n:02d}</div>
  <div class="eyebrow">Part {WORDS[n - 1]}</div>
  <h1>{html.escape(p['title'])}</h1>
  <p class="lead">{html.escape(p['lead'])}</p>
  <ol class="opener-list">{sec_list}</ol>
</section>""")
        prev_short = False
        for i, s in enumerate(p['sections'], 1):
            sid = 'sec-' + slug(s['title'])
            body = decorate(md_to_html(s['body']), n, figs)
            flow = ' flow' if prev_short else ''
            prev_short = len(s['body']) < 1300 and '![' not in s['body']
            out.append(f'<section class="sec{flow}" id="{sid}"><span class="mk">@@S{n}.{i}@@</span><div class="sec-head">'
                       f'<div class="sn">{n}.{i}</div><h2>{html.escape(s["title"])}</h2></div>'
                       f'<div class="body">{body}</div></section>')
        if p['holds']:
            out.append(f'<section class="sec review"><div class="sec-head"><div class="sn">Part {WORDS[n - 1]} in review</div>'
                       f'<h2>What holds, and where to be careful</h2></div>{holds_card(p["holds"])}</section>')
        out.append('</div>')
    # back matter
    out.append('<div class="part back">')
    order = {'Key Insights': 0, 'References': 2}
    fignote = next((s for s in back if s['title'] == 'Figures'), None)
    back = sorted([s for s in back if s['title'] != 'Figures'], key=lambda s: order.get(s['title'], 9))
    for i, s in enumerate(back):
        cls = {'Key Insights': 'insights', 'References': 'refs', 'Figures': 'fignote'}[s['title']]
        mk = '<span class="mk">@@BACK@@</span>' if i == 0 else ''
        body = md_to_html(s['body'])
        if cls == 'insights':
            body = body.replace('<ul>', '<ol class="ins">').replace('</ul>', '</ol>')
            if fignote:
                body += f'<div class="colophon"><div class="lbl">About the figures</div>{md_to_html(fignote["body"])}</div>'
        out.append(f'<section class="sec {cls}">{mk}<div class="sec-head"><div class="sn">'
                   f'{"Summary" if cls == "insights" else "Back matter"}</div><h2>{html.escape(s["title"])}</h2></div>'
                   f'<div class="body">{body}</div></section>')
    out.append('</div>')

    part_pages = ''.join(
        f'@page part{p["num"]} {{ @bottom-left {{ content: "Part {p["num"]} · {p["title"].replace(chr(34), "")}"; }} }}\n'
        for p in parts)
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Circuits and Formation</title>'
            f'<style>{CSS}\n{part_pages}</style></head><body>{"".join(out)}</body></html>')


CSS = """
:root { --ink:#2a2c30; --soft:#5d6470; --faint:#8a919b; --rule:#dcd8cc; --paper:#fbfaf6;
        --accent:#1f5f8b; --accent-soft:#e6eef4; --gold:#a8741a; --gold-bg:#f6efdd;
        --green:#2f7a4f; --green-bg:#ebf3ec; --amber:#9a5a17; --amber-bg:#f8eee2; }
@page { size: Letter; margin: 0.9in 0.85in 0.9in 0.85in; background: var(--paper);
        @bottom-right { content: counter(page); font: 500 8pt 'IBM Plex Sans', sans-serif; color: #8a919b; }
        @bottom-left { content: "Circuits and Formation"; font: 8pt 'IBM Plex Sans', sans-serif; color: #8a919b;
                       letter-spacing: 0.04em; } }
@page cover { margin: 0; @bottom-right { content: none; } @bottom-left { content: none; } }
@page front { @bottom-left { content: "Circuits and Formation"; } }
@page opener { @bottom-right { content: none; } @bottom-left { content: none; } }
html { background: var(--paper); -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { margin: 0; font-family: 'IBM Plex Serif', 'Noto Serif', Georgia, serif; font-size: 10.6pt; line-height: 1.62;
       color: var(--ink); hyphens: auto; font-feature-settings: "kern", "liga"; }
.mk { font-size: 0.5pt; color: var(--paper); position: absolute; }
h1, h2, h3, .eyebrow, .sn, .lbl, figcaption, table, .toc, .opener-list, .cover { font-family: 'IBM Plex Sans', sans-serif; }

/* ---------- cover ---------- */
.cover { page: cover; height: 11in; box-sizing: border-box; padding: 1.35in 1.05in 0.8in 1.05in; position: relative;
         display: flex; flex-direction: column; break-after: page;
         background: linear-gradient(90deg, var(--accent) 0, var(--accent) 0.32in, var(--paper) 0.32in); }
.cover .kicker { font-size: 9.5pt; letter-spacing: 0.06em; color: var(--soft); text-transform: uppercase; }
.cover .kicker em { text-transform: none; font-style: italic; letter-spacing: 0; font-family: 'IBM Plex Serif', serif; font-size: 11pt; }
.cover .title { font-size: 50pt; line-height: 1.02; font-weight: 600; color: #17324a; margin: 0.32in 0 0.14in; letter-spacing: -0.01em; }
.cover .subtitle { font-family: 'IBM Plex Serif', serif; font-style: italic; font-size: 17pt; color: var(--accent); }
.cover .art { width: 100%; margin: 0.65in 0 0.5in; }
.cover-parts { list-style: none; padding: 0; margin: 0; border-top: 1px solid var(--rule); }
.cover-parts li { padding: 7pt 0; border-bottom: 1px solid var(--rule); font-size: 11pt; color: var(--ink); }
.cover-parts li span { display: inline-block; width: 0.45in; color: var(--accent); font-weight: 600; }
.cover .foot { margin-top: auto; font-size: 8.5pt; color: var(--faint); letter-spacing: 0.05em; }

/* ---------- contents ---------- */
.toc { page: front; break-after: page; }
.eyebrow { font-size: 8.5pt; font-weight: 600; letter-spacing: 0.16em; text-transform: uppercase; color: var(--accent); }
.toc .eyebrow { margin-bottom: 10pt; font-size: 10pt; }
.toc-part { display: flex; align-items: baseline; margin: 10pt 0 3pt; padding-bottom: 3pt; border-bottom: 1px solid var(--rule);
            font-weight: 600; font-size: 11.5pt; color: #17324a; }
.toc-part .n { width: 0.42in; color: var(--accent); }
.toc-part .t { flex: 1; }
.toc-part.back { margin-top: 14pt; }
.toc-sec { display: flex; align-items: baseline; font-size: 9.4pt; margin: 1.6pt 0; color: var(--ink); }
.toc-sec .n { width: 0.42in; padding-left: 0.42in; color: var(--faint); font-variant-numeric: tabular-nums; }
.toc-sec .t { flex: 0 1 auto; }
.toc-sec .dots { flex: 1; border-bottom: 1px dotted #c9c4b6; margin: 0 6pt; transform: translateY(-3pt); }
.pg { width: 0.35in; text-align: right; color: var(--soft); font-variant-numeric: tabular-nums; font-weight: 400; }

/* ---------- part opener ---------- */
.part { }
.opener { page: opener; height: 9.15in; display: flex; flex-direction: column; justify-content: center; break-before: page;
          break-after: page; position: relative; }
.opener .big { font-family: 'IBM Plex Sans', sans-serif; font-size: 118pt; font-weight: 200; line-height: 0.9;
               color: #c9d9e6; margin-bottom: 8pt; }
.opener h1 { font-size: 30pt; line-height: 1.12; font-weight: 600; color: #17324a; margin: 6pt 0 10pt; }
.opener .lead { font-style: italic; font-size: 13pt; color: var(--soft); max-width: 5.2in; margin: 0 0 26pt; line-height: 1.5; }
.opener-list { list-style: none; padding: 0; margin: 0; border-top: 2px solid var(--accent); max-width: 5.6in; }
.opener-list li { display: flex; padding: 6.5pt 0; border-bottom: 1px solid var(--rule); font-size: 10.5pt; }
.opener-list .n { width: 0.45in; color: var(--accent); font-weight: 600; }
.opener-list .t { flex: 1; }

/* ---------- sections ---------- */
.sec { break-before: page; }
.sec.flow { break-before: auto; margin-top: 30pt; padding-top: 22pt; border-top: 3px solid #e3dfd3; }
.sec-head { break-after: avoid; }
.p1 .sec { page: part1; } .p2 .sec { page: part2; } .p3 .sec { page: part3; } .p4 .sec { page: part4; } .p5 .sec { page: part5; }
.sec-head { margin: 0 0 18pt; padding-bottom: 10pt; border-bottom: 1px solid var(--rule); }
.sn { font-size: 9pt; font-weight: 600; letter-spacing: 0.12em; color: var(--accent); text-transform: uppercase; margin-bottom: 3pt; }
.sec h2 { font-size: 21pt; line-height: 1.18; font-weight: 600; color: #17324a; margin: 0; letter-spacing: -0.005em; }
.body { max-width: 5.75in; margin: 0 auto 0 0; }
.body h3 { font-size: 11.5pt; color: #17324a; margin: 18pt 0 4pt; }
p { margin: 0 0 9pt; orphans: 3; widows: 3; }
ul, ol { margin: 0 0 10pt; padding-left: 18pt; }
li { margin: 0 0 4pt; }
li::marker { color: var(--accent); }
strong { font-weight: 600; color: #1d2733; }
a { color: var(--accent); text-decoration: none; border-bottom: 1px solid #bcd0e0; }
code { font-family: 'IBM Plex Mono', monospace; font-size: 8.9pt; background: #efece3; padding: 0.5pt 3pt; border-radius: 2pt; }
pre { background: #f2efe6; border-left: 3px solid var(--accent); border-radius: 0 3pt 3pt 0; padding: 9pt 12pt;
      font-size: 8.7pt; line-height: 1.45; break-inside: avoid; margin: 4pt 0 12pt; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 14pt; font-size: 9pt; line-height: 1.4; break-inside: avoid;
        border-top: 1.5px solid #17324a; border-bottom: 1.5px solid #17324a; }
th { text-align: left; font-weight: 600; padding: 6pt 8pt 5pt; border-bottom: 1px solid #17324a; color: #17324a; }
td { padding: 5pt 8pt; border-bottom: 1px solid var(--rule); vertical-align: top; }
tr:last-child td { border-bottom: none; }

/* figures run wider than the text column */
.fig { margin: 12pt 0 14pt; width: 6.8in; break-inside: avoid; }
.fig + p, .fig + ul { break-before: avoid; }
.fig img { max-width: 100%; max-height: 4.3in; width: auto; margin: 0 auto; display: block; border-radius: 3pt; background: white; padding: 6pt; box-sizing: border-box;
           box-shadow: 0 0 0 1px #e4e0d4; }
figcaption { font-size: 8.6pt; line-height: 1.45; color: var(--soft); margin-top: 7pt; max-width: 6.3in; }
figcaption .fl { font-weight: 600; color: var(--accent); margin-right: 6pt; letter-spacing: 0.03em; }

/* callouts */
aside { break-inside: avoid; margin: 12pt 0 14pt; padding: 10pt 14pt 4pt; border-radius: 3pt; }
aside .lbl { font-size: 7.8pt; font-weight: 600; letter-spacing: 0.14em; text-transform: uppercase; margin-bottom: 4pt; }
aside.insight { background: var(--gold-bg); border-left: 3px solid var(--gold); }
aside.insight .lbl { color: var(--gold); }
aside.note { background: var(--accent-soft); border-left: 3px solid var(--accent); }
aside.note .lbl { color: var(--accent); }

/* part review */
.review .sec-head { border-bottom-color: transparent; }
.holds { display: grid; grid-template-columns: 1fr 1fr; gap: 12pt; font-size: 9.8pt; line-height: 1.5; }
.holds .col { padding: 12pt 14pt 6pt; border-radius: 3pt; }
.holds .ok { background: var(--green-bg); border-top: 3px solid var(--green); }
.holds .q { background: var(--amber-bg); border-top: 3px solid var(--amber); }
.holds .lbl { font-size: 8pt; font-weight: 600; letter-spacing: 0.14em; text-transform: uppercase; margin-bottom: 6pt; }
.holds .ok .lbl { color: var(--green); } .holds .q .lbl { color: var(--amber); }
.holds ul { padding-left: 14pt; } .holds li { margin-bottom: 6pt; }

/* back matter */
.back .sec { page: front; }
.insights .body { max-width: none; }
ol.ins { list-style: none; counter-reset: ins; padding: 0; display: grid; grid-template-columns: 1fr 1fr; gap: 10pt 14pt; }
ol.ins li { counter-increment: ins; background: #f3f0e7; border-radius: 3pt; padding: 9pt 11pt 9pt 34pt; position: relative;
            font-size: 9.6pt; line-height: 1.48; margin: 0; break-inside: avoid; }
ol.ins li::before { content: counter(ins, decimal-leading-zero); position: absolute; left: 10pt; top: 8.5pt;
                    font: 600 9pt 'IBM Plex Sans', sans-serif; color: var(--accent); }
.refs .body { max-width: none; columns: 2; column-gap: 0.32in; font-size: 8.1pt; line-height: 1.36;
              font-family: 'IBM Plex Sans', sans-serif; }
.refs .body p { margin: 0 0 4pt; }
.refs .body p strong { display: block; margin-top: 8pt; color: var(--accent); letter-spacing: 0.08em; text-transform: uppercase;
                       font-size: 7.8pt; break-after: avoid; }
.refs .body ul { list-style: none; padding: 0; }
.refs .body li { margin-bottom: 3.5pt; break-inside: avoid; color: var(--ink); }
.refs a { border: none; }
.fignote { break-before: auto; break-inside: avoid; margin-top: 26pt; }
.fignote .sn { display: none; }
.fignote .sec-head { margin-bottom: 8pt; }
.fignote h2 { font-size: 13pt; } .fignote .body { font-size: 9pt; color: var(--soft); }
.front .body { max-width: 5.75in; }
.colophon { margin-top: 22pt; padding-top: 10pt; border-top: 1px solid var(--rule); font-size: 8.8pt; color: var(--soft);
            line-height: 1.5; max-width: 5.75in; break-inside: avoid; }
.colophon .lbl { font-size: 7.8pt; font-weight: 600; letter-spacing: 0.14em; text-transform: uppercase; color: var(--accent);
                 margin-bottom: 4pt; }
"""

# ---------------------------------------------------------------- printing


def chrome():
    c = next((c for c in ['google-chrome', 'chromium', 'chromium-browser'] if shutil.which(c)), None)
    if not c:
        sys.exit('No Chrome/Chromium found to print the PDF.')
    return c


def print_pdf(html_text, out_pdf):
    with tempfile.NamedTemporaryFile('w', suffix='.html', dir='.', delete=False, encoding='utf-8') as f:
        f.write(html_text)
        path = f.name
    try:
        subprocess.run([chrome(), '--headless=new', '--disable-gpu', '--no-pdf-header-footer',
                        '--allow-file-access-from-files', f'--print-to-pdf={out_pdf}', 'file://' + os.path.abspath(path)],
                       check=True, capture_output=True)
    finally:
        os.remove(path)


def find_pages(pdf):
    if not shutil.which('pdftotext'):
        return {}
    txt = subprocess.run(['pdftotext', pdf, '-'], capture_output=True, text=True, check=True).stdout
    pages = {}
    for i, page in enumerate(txt.split('\f'), 1):
        for key in re.findall(r'@@(\w[\w.]*)@@', page):
            pages.setdefault(key, i)
    return pages


def main():
    pre, parts, back = parse(open(SRC, encoding='utf-8').read())
    print_pdf(build_html(pre, parts, back, {}), OUT_PDF)          # pass 1: layout
    pages = find_pages(OUT_PDF)
    print_pdf(build_html(pre, parts, back, pages), OUT_PDF)       # pass 2: with page numbers
    print(f'Saved {OUT_PDF} ({len(pages)} anchors located)')


if __name__ == '__main__':
    main()
