import re

with open('app.py', encoding='utf-8') as f:
    src = f.read()

# ── 1. render_disclaimer: HTML div → st.warning() ────────────────────────────
old = r'def render_disclaimer\(\):\n    st\.markdown\("""[\s\S]+?""", unsafe_allow_html=True\)'
new = (
    'def render_disclaimer():\n'
    '    st.warning(\n'
    '        "**Research / Educational Prototype \u2014 Not for Clinical Diagnosis or Treatment:** "\n'
    '        "Predictions and analytics are statistical estimates from CDC & FDA NARMS surveillance data. "\n'
    '        "They must not replace antimicrobial susceptibility testing (AST), clinical microbiology, "\n'
    '        "or qualified medical judgment. This system does not prescribe or select antimicrobial therapy."\n'
    '    )'
)
src = re.sub(old, new, src)

# ── 2. main-header divs → st.title() ─────────────────────────────────────────
src = re.sub(
    r"st\.markdown\('(<div class=\"main-header\">(.+?)</div>)', unsafe_allow_html=True\)",
    lambda m: f'st.title("{m.group(2)}")',
    src
)
src = re.sub(
    r'st\.markdown\(f\'<div class="main-header">(.+?)</div>\', unsafe_allow_html=True\)',
    lambda m: f'st.title(f"{m.group(1)}")',
    src
)

# ── 3. sub-header divs → st.caption() ────────────────────────────────────────
src = re.sub(
    r"st\.markdown\('(<div class=\"sub-header\">(.+?)</div>)', unsafe_allow_html=True\)",
    lambda m: f'st.caption("{m.group(2)}")',
    src
)
src = re.sub(
    r'st\.markdown\(f\'<div class="sub-header">(.+?)</div>\', unsafe_allow_html=True\)',
    lambda m: f'st.caption(f"{m.group(1)}")',
    src
)

# ── 4. safety-box (static) → st.info() ───────────────────────────────────────
SAFETY_REPLACEMENT = 'st.info("**Clinical Safety Notice:** Potential option for clinical review \u2014 requires clinical confirmation.")'
src = re.sub(
    r"st\.markdown\('(<div class=\"safety-box\">)(<strong>Clinical Safety Notice:</strong>[^<]*)</div>', unsafe_allow_html=True\)",
    SAFETY_REPLACEMENT,
    src
)
# safety-box f-string variant
src = src.replace(
    """st.markdown(f'<div class="safety-box">{profile["safety_guidance"]}</div>', unsafe_allow_html=True)""",
    'st.info(profile["safety_guidance"])'
)

# ── 5. status badges → st.error() / st.success() ─────────────────────────────
# Prediction page badges (f-string)
src = re.sub(
    r'st\.markdown\(f\'<div class="status-badge-res">PREDICTED: \{pred\["prediction"\]\.upper\(\)\}</div>\', unsafe_allow_html=True\)',
    """st.error(f"PREDICTED: {pred['prediction'].upper()} (Resistant)")""",
    src
)
src = re.sub(
    r'st\.markdown\(f\'<div class="status-badge-susc">PREDICTED: \{pred\["prediction"\]\.upper\(\)\}</div>\', unsafe_allow_html=True\)',
    """st.success(f"PREDICTED: {pred['prediction'].upper()} (Susceptible)")""",
    src
)

# Anomaly page — static span badges
src = src.replace(
    """st.markdown('<span class="status-badge-res">Unusual Sample</span>', unsafe_allow_html=True)""",
    'st.error("Unusual Sample")'
)
src = src.replace(
    """st.markdown('<span class="status-badge-susc">Standard Sample</span>', unsafe_allow_html=True)""",
    'st.success("Standard Sample")'
)

# Resistance Profile — MDR badges
src = src.replace(
    """st.markdown('<div class="status-badge-res">MDR Alert Detected</div>', unsafe_allow_html=True)""",
    'st.error("MDR Alert Detected")'
)
src = src.replace(
    """st.markdown('<div class="status-badge-susc">Standard Resistance Profile</div>', unsafe_allow_html=True)""",
    'st.success("Standard Resistance Profile")'
)

# Anomaly detection status (f-string)
src = re.sub(
    r'st\.markdown\(f\'<div class="status-badge-res">STATUS: \{res_an\["status"\]\.upper\(\)\}</div>\', unsafe_allow_html=True\)',
    """st.error(f"STATUS: {res_an['status'].upper()}")""",
    src
)
src = re.sub(
    r'st\.markdown\(f\'<div class="status-badge-susc">STATUS: \{res_an\["status"\]\.upper\(\)\}</div>\', unsafe_allow_html=True\)',
    """st.success(f"STATUS: {res_an['status'].upper()}")""",
    src
)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(src)

# Verify — print any remaining custom HTML lines
remaining = []
for i, line in enumerate(src.splitlines(), 1):
    if any(x in line for x in ['unsafe_allow_html', 'status-badge', 'main-header', 'sub-header', 'disclaimer-box', 'safety-box']):
        remaining.append(f"  L{i}: {line.strip()}")

if remaining:
    print("REMAINING custom HTML lines:")
    for r in remaining:
        print(r)
else:
    print("Clean — no custom HTML classes remaining.")
