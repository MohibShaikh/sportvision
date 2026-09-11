"""Editorial night-edition styling for the review app.

The app is set like a match report printed white on slate. A serif carries the
headlines and figures, hairline rules do the work cards would normally do, and
pale gold is both the accent and the ball, so the one warm mark on the page is
always the thing you are looking for. The two team inks do the rest.

Widget selectors here track Streamlit's ``data-testid`` attributes, so they may
need revisiting on a major Streamlit upgrade. Streamlit's own heading and
paragraph rules outrank a bare class, so the editorial elements are
element-qualified and name their font face explicitly.
"""

import html

import streamlit as st

# sRGB equivalents of the OKLCH values in STYLES. The Streamlit CLI only
# accepts hex, so cli.py passes these as --theme.* flags to keep the base
# widget chrome in step with the stylesheet below.
PALETTE = {
    "bg": "#1D2027",
    "bg_2": "#292D34",
    "fg": "#EBE7E0",
    "accent": "#D9B568",
    "team_1": "#67A2D2",
    "team_2": "#D66D5E",
    "ball": "#D9B568",
    "unassigned": "#5E646B",
}

STYLES = """<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;700&family=Newsreader:ital,opsz,wght@0,6..72,300..700;1,6..72,300..500&display=swap');

:root {
    /* Flat slate stock. Nothing here is pure black or pure white. */
    --bg: oklch(0.245 0.013 265);
    --bg-2: oklch(0.295 0.014 262);
    --bg-3: oklch(0.345 0.014 260);
    --fg: oklch(0.930 0.010 85);
    --fg-2: oklch(0.795 0.012 250);
    --fg-3: oklch(0.680 0.014 250);
    --rule: oklch(0.375 0.014 260);
    --rule-strong: oklch(0.500 0.014 258);
    /* The accent is the ball. It is never spent on decoration. */
    --accent: oklch(0.790 0.105 85);
    --accent-hi: oklch(0.860 0.100 85);
    --team-1: oklch(0.690 0.095 245);
    --team-2: oklch(0.655 0.135 30);

    --serif: 'Newsreader', 'Iowan Old Style', 'Source Serif 4', Georgia, serif;
    --sans: 'Archivo', 'Helvetica Neue', Helvetica, sans-serif;

    /* Deliberately uneven. Spacing separates the sections, so no box has to. */
    --gap-1: 0.35rem;
    --gap-2: 0.7rem;
    /* Streamlit sizes columns with calc(X% - 16px), which assumes a 32px gap.
       Anything wider than that overflows the row and wraps it. */
    --gap-3: clamp(1.1rem, 0.9rem + 0.6vw, 1.6rem);
    --gap-4: clamp(2rem, 1.4rem + 2vw, 3.4rem);
    --gap-5: clamp(2.6rem, 1.8rem + 2.6vw, 4.2rem);
}

/* ---------- canvas ---------- */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background: var(--bg);
}
[data-testid="stHeader"] {
    background: transparent;
}
[data-testid="stAppDeployButton"] {
    display: none;
}
[data-testid="stMainBlockContainer"] {
    /* Wide layout for the review frame, but the text keeps a real measure. */
    max-width: 70rem;
    padding: var(--gap-4) var(--gap-3) var(--gap-5);
}
body,
.stMarkdown,
[data-testid="stMarkdownContainer"] p,
label,
button,
input,
textarea,
select {
    font-family: var(--sans);
    color: var(--fg);
}
[data-testid="stMarkdownContainer"] p {
    color: var(--fg-2);
    font-size: 0.95rem;
    line-height: 1.62;
    max-width: 64ch;
}
[data-testid="stCaptionContainer"] p {
    color: var(--fg-3);
    font-size: 0.815rem;
    line-height: 1.55;
    max-width: 70ch;
}

/* ---------- masthead ---------- */
.sv-masthead {
    border-top: 3px solid var(--fg);
    padding-top: 0.55rem;
    margin-bottom: var(--gap-4);
}
.sv-flag {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: var(--gap-3);
    flex-wrap: wrap;
    border-bottom: 1px solid var(--rule);
    padding-bottom: 0.5rem;
}
.sv-wordmark {
    font-family: var(--serif);
    font-size: 1.32rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    color: var(--fg);
}
.sv-wordmark em {
    font-style: italic;
    font-weight: 300;
    color: var(--accent);
}
.sv-dateline {
    font-family: var(--sans);
    font-size: 0.665rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--fg-3);
}
h1.sv-headline {
    font-family: var(--serif);
    font-size: clamp(2.5rem, 1.5rem + 3.4vw, 4.35rem);
    font-weight: 500;
    line-height: 0.98;
    letter-spacing: -0.022em;
    color: var(--fg);
    margin: var(--gap-3) 0 0;
    max-width: 17ch;
}
h1.sv-headline em {
    font-style: italic;
    font-weight: 300;
}
p.sv-standfirst {
    /* Indented off the headline rather than centered under it. */
    margin: var(--gap-3) 0 0 clamp(0rem, -3rem + 12vw, 9rem);
    max-width: 44ch;
    border-left: 2px solid var(--accent);
    padding-left: var(--gap-2);
    font-size: 1.005rem;
    line-height: 1.55;
    color: var(--fg-2);
}
p.sv-standfirst b {
    font-weight: 600;
    color: var(--fg);
}

/* ---------- section marks ---------- */
.sv-section {
    display: flex;
    align-items: baseline;
    gap: var(--gap-2);
    border-top: 1px solid var(--fg);
    padding-top: 0.45rem;
    margin: var(--gap-5) 0 var(--gap-3);
}
.sv-section-no {
    font-family: var(--serif);
    font-size: 0.92rem;
    font-weight: 600;
    color: var(--accent);
    font-variant-numeric: tabular-nums;
    flex: none;
}
.sv-section h2 {
    font-family: var(--serif);
    font-size: clamp(1.3rem, 1.1rem + 0.7vw, 1.72rem);
    font-weight: 500;
    letter-spacing: -0.012em;
    color: var(--fg);
    margin: 0;
    padding: 0;
    line-height: 1.2;
}
.sv-section .sv-kicker {
    font-family: var(--sans);
    margin-left: auto;
    font-size: 0.665rem;
    font-weight: 600;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--fg-3);
    flex: none;
}

.sv-subhead {
    display: flex;
    align-items: baseline;
    gap: var(--gap-2);
    border-top: 1px solid var(--rule-strong);
    padding-top: 0.4rem;
    margin: var(--gap-4) 0 var(--gap-2);
}
.sv-subhead h3 {
    font-family: var(--serif);
    font-style: italic;
    font-size: 1.22rem;
    font-weight: 400;
    color: var(--fg);
    margin: 0;
    padding: 0;
    line-height: 1.25;
}
/* Scoped to the class. A bare `span` also matches the span Streamlit nests
   inside headings, which restyles the title itself. */
.sv-subhead .sv-kicker {
    font-family: var(--sans);
    margin-left: auto;
    font-size: 0.665rem;
    font-weight: 600;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--fg-3);
    flex: none;
}

/* ---------- standing figures ---------- */
.sv-figures {
    display: flex;
    flex-wrap: wrap;
    gap: var(--gap-3) var(--gap-4);
    margin: var(--gap-3) 0 var(--gap-2);
}
.sv-figure {
    border-top: 1px solid var(--rule-strong);
    padding-top: 0.4rem;
    min-width: 8rem;
    flex: 1 1 8rem;
}
.sv-figure:first-child {
    /* One figure leads. Equal weight everywhere reads as a dashboard. */
    flex-basis: 12rem;
    border-top-color: var(--fg);
}
.sv-figure-n {
    font-family: var(--serif);
    display: block;
    font-size: 2.15rem;
    font-weight: 500;
    line-height: 1.05;
    letter-spacing: -0.02em;
    color: var(--fg);
    font-variant-numeric: tabular-nums lining-nums;
}
.sv-figure:first-child .sv-figure-n {
    font-size: 2.95rem;
}
.sv-figure-l {
    font-family: var(--sans);
    display: block;
    margin-top: 0.28rem;
    font-size: 0.735rem;
    font-weight: 500;
    letter-spacing: 0.055em;
    color: var(--fg-3);
    line-height: 1.35;
}

/* ---------- possession, set as a ruled table ---------- */
.sv-share {
    display: grid;
    grid-template-columns: minmax(5.5rem, auto) 1fr minmax(3.6rem, auto);
    align-items: center;
    gap: var(--gap-2);
    border-bottom: 1px solid var(--rule);
    padding: 0.62rem 0;
}
.sv-share:first-of-type {
    border-top: 1px solid var(--fg);
}
.sv-share-team {
    font-family: var(--sans);
    font-size: 0.815rem;
    font-weight: 600;
    letter-spacing: 0.055em;
    text-transform: uppercase;
    color: var(--fg-2);
    display: flex;
    align-items: center;
    gap: 0.45rem;
}
.sv-share-swatch {
    width: 0.62rem;
    height: 0.62rem;
    flex: none;
    background: var(--swatch);
}
/* Spans are inline, so both the track and the fill need an explicit box or
   their width and height are ignored and the bar never draws. */
.sv-share-track {
    display: block;
    height: 3px;
    background: var(--rule);
}
.sv-share-fill {
    display: block;
    height: 100%;
    background: var(--swatch);
    transform-origin: left center;
    animation: sv-wipe 0.85s cubic-bezier(0.22, 1, 0.36, 1) both;
}
.sv-share-n {
    font-family: var(--serif);
    font-size: 1.62rem;
    font-weight: 500;
    line-height: 1;
    text-align: right;
    color: var(--fg);
    font-variant-numeric: tabular-nums lining-nums;
}

/* ---------- footnotes and colophon ---------- */
p.sv-note {
    border-left: 1px solid var(--rule-strong);
    padding-left: var(--gap-2);
    margin: var(--gap-2) 0;
    font-size: 0.815rem;
    line-height: 1.55;
    color: var(--fg-3);
    max-width: 66ch;
}
.sv-guide {
    margin: var(--gap-3) 0 0;
    max-width: 58ch;
}
.sv-guide-item {
    display: flex;
    gap: var(--gap-2);
    border-top: 1px solid var(--rule);
    padding: 0.72rem 0;
}
.sv-guide-item span:first-child {
    font-family: var(--serif);
    font-size: 0.9rem;
    color: var(--accent);
    font-variant-numeric: tabular-nums;
    flex: none;
    padding-top: 0.08rem;
}
.sv-guide-item span:last-child {
    font-family: var(--sans);
    font-size: 0.925rem;
    line-height: 1.55;
    color: var(--fg-2);
}
.sv-colophon {
    font-family: var(--sans);
    border-top: 3px solid var(--fg);
    margin-top: var(--gap-5);
    padding-top: 0.55rem;
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem var(--gap-3);
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    color: var(--fg-3);
}

/* ---------- controls ---------- */
[data-testid="stBaseButton-primary"] {
    background: var(--fg);
    color: var(--bg);
    border: 1px solid var(--fg);
    border-radius: 2px;
    font-weight: 600;
    font-size: 0.875rem;
    letter-spacing: 0.02em;
    padding: 0.48rem 1.6rem;
    transition: background 0.18s cubic-bezier(0.22, 1, 0.36, 1),
        border-color 0.18s cubic-bezier(0.22, 1, 0.36, 1);
}
[data-testid="stBaseButton-primary"]:hover:not(:disabled) {
    background: var(--accent);
    border-color: var(--accent);
    color: var(--bg);
}
[data-testid="stBaseButton-secondary"],
.stDownloadButton button {
    background: transparent;
    color: var(--fg);
    border: 1px solid var(--rule-strong);
    border-radius: 2px;
    font-weight: 500;
    font-size: 0.875rem;
    padding: 0.48rem 1.15rem;
    transition: border-color 0.18s cubic-bezier(0.22, 1, 0.36, 1),
        color 0.18s cubic-bezier(0.22, 1, 0.36, 1);
}
[data-testid="stBaseButton-secondary"]:hover:not(:disabled),
.stDownloadButton button:hover {
    border-color: var(--fg);
    color: var(--accent);
}
[data-testid="stBaseButton-primary"]:disabled,
[data-testid="stBaseButton-secondary"]:disabled {
    opacity: 0.45;
}
[data-testid="stBaseLinkButton-secondary"] {
    border: 1px solid var(--rule-strong);
    border-radius: 2px;
    color: var(--fg);
}
a,
[data-testid="stMarkdownContainer"] a {
    color: var(--accent);
    text-decoration-thickness: 1px;
    text-underline-offset: 0.18em;
}
a:hover,
[data-testid="stMarkdownContainer"] a:hover {
    color: var(--fg);
}
.sv-colophon a {
    color: var(--fg-2);
}
.sv-colophon a:hover {
    color: var(--accent);
}

label,
[data-testid="stWidgetLabel"] p {
    font-size: 0.73rem;
    font-weight: 600;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: var(--fg-3);
}
[data-baseweb="select"] > div,
[data-baseweb="input"],
[data-testid="stTextInput"] input {
    background: var(--bg-2);
    border: 1px solid var(--rule);
    border-radius: 2px;
    color: var(--fg);
}
[data-baseweb="select"] > div:focus-within,
[data-baseweb="input"]:focus-within {
    border-color: var(--accent);
    box-shadow: none;
}
[data-baseweb="popover"] [role="listbox"] {
    background: var(--bg-2);
    border: 1px solid var(--rule-strong);
    border-radius: 2px;
}
[role="option"]:hover {
    background: var(--bg-3);
}
[data-testid="stFileUploaderDropzone"] {
    background: var(--bg-2);
    border: 1px solid var(--rule-strong);
    border-radius: 2px;
    padding: var(--gap-3);
    transition: border-color 0.18s cubic-bezier(0.22, 1, 0.36, 1);
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--accent);
}
[data-testid="stFileUploaderDropzoneInstructions"] span {
    color: var(--fg-3);
}
[data-testid="stColorBlock"],
[data-testid="stColorPicker"] > div > div {
    border-radius: 2px;
    border: 1px solid var(--rule-strong);
}
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: var(--accent);
    border-color: var(--accent);
}
[data-testid="stProgress"] > div > div > div > div {
    background: var(--accent);
}
[data-testid="stProgress"] > div > div > div {
    background: var(--rule);
}

/* Expanders carry rules, not fills. Nothing here is a card. */
[data-testid="stExpander"] details {
    background: transparent;
    border: 0;
    border-top: 1px solid var(--rule);
    border-radius: 0;
}
[data-testid="stExpander"] summary {
    background: transparent;
    border-radius: 0;
    padding: 0.62rem 0;
    font-size: 0.73rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--fg-2);
}
[data-testid="stExpander"] summary:hover {
    color: var(--accent);
}
[data-testid="stExpanderDetails"] {
    padding: 0 0 var(--gap-2);
}

[data-testid="stAlert"] {
    background: var(--bg-2);
    border: 0;
    border-left: 2px solid var(--rule-strong);
    border-radius: 0;
    padding: 0.72rem var(--gap-2);
    color: var(--fg-2);
}
[data-testid="stAlert"] p {
    font-size: 0.885rem;
    color: var(--fg-2);
}
[data-testid="stAlertContentWarning"],
[data-testid="stAlertContentError"] {
    border-left-color: var(--accent);
}
[data-testid="stAlert"] svg {
    display: none;
}

/* Scoped to the group. Unscoped, this also restyled the widget's own label. */
[data-testid="stRadio"] [role="radiogroup"] label p {
    font-size: 0.9rem;
    font-weight: 400;
    letter-spacing: 0;
    text-transform: none;
    color: var(--fg-2);
}
[data-testid="stRadio"] [role="radiogroup"] {
    gap: 0.1rem;
}
[data-testid="stHorizontalBlock"] {
    gap: var(--gap-3);
}
[data-testid="stHorizontalBlock"] [data-testid="stHorizontalBlock"] {
    gap: var(--gap-2);
}
hr {
    border-color: var(--rule);
}
::-webkit-scrollbar {
    width: 9px;
    height: 9px;
}
::-webkit-scrollbar-track {
    background: var(--bg);
}
::-webkit-scrollbar-thumb {
    background: var(--rule-strong);
}

/* ---------- motion: one orchestrated arrival, then stillness ---------- */
@keyframes sv-rise {
    from {
        opacity: 0;
        transform: translateY(9px);
    }
    to {
        opacity: 1;
        transform: none;
    }
}
@keyframes sv-wipe {
    from {
        transform: scaleX(0);
    }
    to {
        transform: scaleX(1);
    }
}
.sv-masthead {
    animation: sv-rise 0.62s cubic-bezier(0.22, 1, 0.36, 1) both;
}
h1.sv-headline {
    animation: sv-rise 0.62s cubic-bezier(0.22, 1, 0.36, 1) 0.07s both;
}
p.sv-standfirst {
    animation: sv-rise 0.62s cubic-bezier(0.22, 1, 0.36, 1) 0.14s both;
}
.sv-guide-item {
    animation: sv-rise 0.55s cubic-bezier(0.22, 1, 0.36, 1) both;
}
.sv-guide-item:nth-child(2) {
    animation-delay: 0.07s;
}
.sv-guide-item:nth-child(3) {
    animation-delay: 0.14s;
}
@media (prefers-reduced-motion: reduce) {
    *,
    *::before,
    *::after {
        animation-duration: 0.001ms !important;
        animation-delay: 0ms !important;
        transition-duration: 0.001ms !important;
    }
}
</style>"""


def apply():
    st.markdown(STYLES, unsafe_allow_html=True)


def _write(markup):
    st.markdown(markup, unsafe_allow_html=True)


def masthead(dateline, headline, standfirst):
    _write(
        '<div class="sv-masthead"><div class="sv-flag">'
        '<span class="sv-wordmark">Sport<em>Vision</em></span>'
        f'<span class="sv-dateline">{html.escape(dateline)}</span>'
        f'</div></div><h1 class="sv-headline">{headline}</h1>'
        f'<p class="sv-standfirst">{standfirst}</p>'
    )


def section(number, title, kicker=None):
    mark = f'<span class="sv-kicker">{html.escape(kicker)}</span>' if kicker else ""
    _write(
        f'<div class="sv-section"><span class="sv-section-no">{number:02d}</span>'
        f"<h2>{html.escape(title)}</h2>{mark}</div>"
    )


def subhead(title, kicker=None):
    mark = f'<span class="sv-kicker">{html.escape(kicker)}</span>' if kicker else ""
    _write(f'<div class="sv-subhead"><h3>{html.escape(title)}</h3>{mark}</div>')


def figures(items):
    """items: sequence of (value, label). The first one is set larger."""
    cells = "".join(
        f'<div class="sv-figure"><span class="sv-figure-n">{html.escape(str(v))}'
        f'</span><span class="sv-figure-l">{html.escape(label)}</span></div>'
        for v, label in items
    )
    _write(f'<div class="sv-figures">{cells}</div>')


def shares(rows):
    """rows: sequence of (team label, fraction 0-1, ink)."""
    body = "".join(
        f'<div class="sv-share" style="--swatch:{ink}">'
        f'<span class="sv-share-team"><span class="sv-share-swatch"></span>'
        f"{html.escape(label)}</span>"
        f'<span class="sv-share-track"><span class="sv-share-fill" '
        f'style="width:{share:.1%}"></span></span>'
        f'<span class="sv-share-n">{share:.0%}</span></div>'
        for label, share, ink in rows
    )
    _write(body)


def note(text):
    _write(f'<p class="sv-note">{html.escape(text)}</p>')


def guide(items):
    body = "".join(
        f'<div class="sv-guide-item"><span>{i:02d}</span>'
        f"<span>{html.escape(text)}</span></div>"
        for i, text in enumerate(items, 1)
    )
    _write(f'<div class="sv-guide">{body}</div>')


def colophon(parts):
    body = "".join(f"<span>{part}</span>" for part in parts)
    _write(f'<div class="sv-colophon">{body}</div>')
