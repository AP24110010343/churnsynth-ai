# ChurnSynth Design System Specification
**Version:** 2.0.0-PRO  
**Classification:** Enterprise Quantitative Analytics & Generative AI Systems  
**Design Caliber:** Linear (Precision) × Vercel (Typographic Restraint) × Bloomberg/FactSet (Information Density)  
**Target Environment:** Streamlit / Web Client (`churnsynth_ai`)

---

## 1. Executive Summary & Design Philosophy

The ChurnSynth Design System establishes the visual and architectural standards for high-fidelity synthetic data generation and predictive ML benchmarking. Designed for quantitative researchers, ML engineers, and enterprise risk officers, this system rejects consumer-grade AI clichés in favor of institutional-grade operational software.

### 1.1 The Anti-Trope Mandate

```
┌───────────────────────────────────────┬──────────────────────────────────────┐
│ REJECTED (Generic AI Clichés)         │ ADOPTED (Institutional Precision)   │
├───────────────────────────────────────┼──────────────────────────────────────┤
│ Ambient glowing background blobs      │ Solid matte neutral canvas (#080A0F) │
│ Rainbow neon gradients (#FF00FF)      │ Monochromatic baseline + dual data   │
│ Puffy rounded corners (14px – 24px)   │ Strict 4px / 6px micro-radii         │
│ Low-density cards with empty space    │ High-density information ribbons     │
│ Cartoon emojis in alerts (💡, 🚀, 🤖) │ Unicode micro-glyphs (◈, ↗, ↘, ⬡, ℹ) │
│ Fluffy button glows & animated pulses │ Crisp 1px borders, subtle transitions│
│ Center-aligned conversational text    │ Left-aligned structured data grids   │
│ Inconsistent variable-width numbers   │ Tabular figures (tnum, zero-slash)   │
└───────────────────────────────────────┴──────────────────────────────────────┘
```

### 1.2 Core Principles

1. **Information Density with Absolute Legibility**: Every square pixel must serve an analytical purpose. Whitespace is used rhythmically (4px / 8px scale) to organize density, never to create hollow padding.
2. **Deterministic Hierarchy**: Surfaces transition logically from the background canvas through elevated analytical modules. Borders are sharp 1px physical divisions, not fuzzy drop shadows.
3. **Tabular Numerics**: All numbers, statistics, metrics, and timestamps MUST render with fixed-width tabular digits (`tnum`) to ensure numerical columns and deltas align vertically.
4. **Subdued Semantic Coloration**: Color is reserved exclusively for state and data differentiation. Backgrounds remain strictly neutral dark slate/carbon.

---

## 2. Complete Design Token Matrix

### 2.1 Surface & Layer Hierarchy

The application employs a 4-tier surface depth model. Rather than relying on heavy elevation shadows, layers are differentiated via subtle background shifts and crisp 1px borders.

```
┌─────────────────────────────────────────────────────────────┐
│ Surface L3: Active States, Inset Wells, Tab Highlight       │
│ #1E2638  |  border: rgba(255, 255, 255, 0.12)              │
├─────────────────────────────────────────────────────────────┤
│ Surface L2: Elevated Modules, Modal Dialogs, Tooltips, Hover│
│ #161C2A  |  border: rgba(255, 255, 255, 0.09)              │
├─────────────────────────────────────────────────────────────┤
│ Surface L1: Primary Content Panels, Metric Cards, Sidebar    │
│ #0F131D  |  border: rgba(255, 255, 255, 0.07)              │
├─────────────────────────────────────────────────────────────┤
│ Canvas / Root: Background Atmosphere                        │
│ #080A0F  |  border: none                                    │
└─────────────────────────────────────────────────────────────┘
```

#### Exact Hex & Alpha Tokens

| Token Name | Value | Purpose |
| :--- | :--- | :--- |
| `--surface-canvas` | `#080A0F` | Global page background; matte, non-distracting canvas |
| `--surface-sidebar` | `#0A0D15` | Fixed navigation & parameter control panel |
| `--surface-l1` | `#0F131D` | Primary card background, metric containers, table wrappers |
| `--surface-l2` | `#161C2A` | Elevated interactive controls, popovers, table hover rows |
| `--surface-l3` | `#1E2638` | Active tab background, selected state, inset wells |
| `--border-subtle` | `rgba(255, 255, 255, 0.05)` | Inner card dividers, sub-row separators |
| `--border-default` | `rgba(255, 255, 255, 0.07)` | Standard module boundary, table border, card outline |
| `--border-hover` | `rgba(255, 255, 255, 0.12)` | Interactive element hover state |
| `--border-active` | `rgba(255, 255, 255, 0.20)` | Focused inputs, selected items, active button border |

---

### 2.2 Typography System

The typography system pairs **Plus Jakarta Sans** (or **Inter**) for structured headings and UI controls with **JetBrains Mono** for numerical values, IDs, mathematical formulas, and status tags.

#### Typography Scale Matrix

| Role | Font Family | Size | Weight | Line Height | Letter Spacing | Transform |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Hero Display** | Primary Sans | 22px (`1.375rem`) | 600 (SemiBold) | 28px (`1.27`) | `-0.025em` | None |
| **Section Title** | Primary Sans | 15px (`0.9375rem`)| 600 (SemiBold) | 20px (`1.33`) | `-0.015em` | None |
| **Card Header** | Primary Sans | 13px (`0.8125rem`)| 600 (SemiBold) | 18px (`1.38`) | `-0.010em` | None |
| **Body Default** | Primary Sans | 13px (`0.8125rem`)| 400 (Regular) | 19px (`1.46`) | `0.000em` | None |
| **Body Compact** | Primary Sans | 12px (`0.75rem`) | 400 (Regular) | 16px (`1.33`) | `0.005em` | None |
| **Micro Caption** | Primary Sans | 11px (`0.6875rem`)| 400 (Regular) | 15px (`1.36`) | `0.010em` | None |
| **Metric Large** | JetBrains Mono | 24px (`1.5rem`) | 600 (SemiBold) | 28px (`1.16`) | `-0.020em` | Tabular Nums |
| **Metric Medium**| JetBrains Mono | 18px (`1.125rem`)| 600 (SemiBold) | 22px (`1.22`) | `-0.015em` | Tabular Nums |
| **Data Cell** | JetBrains Mono | 12px (`0.75rem`) | 400 / 500 | 16px (`1.33`) | `0.000em` | Tabular Nums |
| **Meta Label** | JetBrains Mono | 10px (`0.625rem`)| 600 (SemiBold) | 13px (`1.30`) | `+0.060em` | Uppercase |
| **Status Tag** | JetBrains Mono | 10.5px (`0.656rem`)| 500 (Medium) | 14px (`1.33`) | `+0.020em` | None |

#### Mandatory OpenType Font Features
```css
/* All numeric & data components must enforce: */
font-feature-settings: "cv02", "cv03", "cv04", "cv11", "tnum" 1, "zero" 1;
font-variant-numeric: tabular-nums slashed-zero;
```

---

### 2.3 Text & Foreground Tokens

| Token Name | Hex Value | Purpose / Application |
| :--- | :--- | :--- |
| `--text-primary` | `#F8FAFC` | Primary metrics, headings, high-contrast labels, active state |
| `--text-secondary`| `#94A3B8` | Subtitles, descriptive copy, inactive tabs, table header labels |
| `--text-tertiary` | `#64748B` | Metric micro-labels, timestamps, metadata keys, border guides |
| `--text-disabled` | `#334155` | Deactivated controls, null values, unavailable actions |

---

### 2.4 Accent & Semantic Tokens

Semantic colors are calibrated for dark backgrounds, maintaining a minimum contrast ratio of 4.5:1 against Surface L1 (`#0F131D`). Background tints use 8% opacity with matching 20% alpha borders.

| Semantic Role | Primary Hex | Soft Bg (8% Alpha) | Border (20% Alpha) | Application |
| :--- | :--- | :--- | :--- | :--- |
| **Brand / Action** | `#4F46E5` | `rgba(79, 70, 229, 0.12)` | `rgba(99, 102, 241, 0.35)` | Primary CTAs, active sampling, model focus |
| **Data Real / GT** | `#6366F1` | `rgba(99, 102, 241, 0.08)` | `rgba(99, 102, 241, 0.20)` | Real baseline cohort, empirical distribution |
| **Data Synth / DDPM**| `#10B981`| `rgba(16, 185, 129, 0.08)` | `rgba(16, 185, 129, 0.20)` | Synthetic cohort, diffusion outputs |
| **Positive Delta** | `#10B981` | `rgba(16, 185, 129, 0.08)` | `rgba(16, 185, 129, 0.20)` | Accurate churn parity, high ML AUROC |
| **Negative Delta** | `#F43F5E` | `rgba(244, 63, 94, 0.08)` | `rgba(244, 63, 94, 0.20)` | High churn discrepancy, distribution drift |
| **Warning / Skew** | `#F59E0B` | `rgba(245, 158, 11, 0.08)` | `rgba(245, 158, 11, 0.20)` | Moderate KS divergence, tail imbalance |
| **Info / Protocol** | `#0EA5E9` | `rgba(14, 165, 233, 0.08)` | `rgba(14, 165, 233, 0.20)` | Architectural specs, protocol definitions |

---

### 2.5 Spatial System & Border Radius Scale

#### Spatial Grid Scale (4px base)
- `space-1`: `4px` (Inline item separation, status dot spacing)
- `space-2`: `8px` (Badge padding, input internal vertical padding)
- `space-3`: `12px` (Card inner padding compact, metric card gutters)
- `space-4`: `16px` (Standard card padding, tab navigation margin)
- `space-5`: `20px` (Section gaps, major module row spacing)
- `space-6`: `24px` (Major container separation)
- `space-8`: `32px` (Page boundary margins)

#### Border Radius Discipline
```
┌────────────────────────────────────────────────────────┐
│ Radius 3px: Small inline badges, status dots, tags     │
├────────────────────────────────────────────────────────┤
│ Radius 4px: Input fields, dropdown triggers, mono pills│
├────────────────────────────────────────────────────────┤
│ Radius 6px: Buttons, segmented tabs, KPI cards, tables │
├────────────────────────────────────────────────────────┤
│ Radius 8px: Master container panels (Maximum allowed)  │
└────────────────────────────────────────────────────────┘
```
*Rule: Never exceed 8px border-radius anywhere in the application.*

---

## 3. Component Blueprints & Technical Specifications

---

### 3.1 Top Navigation & Status Breadcrumbs Bar

A streamlined horizontal header providing instant system orientation without consuming vertical real estate.

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│  ◈ CHURNSYNTH  /  RETAIL ANALYTICS  /  SUBSCRIPTION CHURN SIMULATOR     [● READY] │
└──────────────────────────────────────────────────────────────────────────────────┘
```

#### Technical Specification
- **Height:** 42px
- **Border Bottom:** `1px solid rgba(255, 255, 255, 0.07)`
- **Breadcrumb Font:** `JetBrains Mono`, 11px (`0.6875rem`), weight 500
- **Breadcrumb Separator:** `#334155` (`/`)
- **Active Node:** `#F8FAFC`, weight 600
- **Status Indicator:**
  - Pill: Background `#0F131D`, border `1px solid rgba(255, 255, 255, 0.08)`, border-radius 3px, padding `2px 8px`
  - Solid Dot: `5px × 5px`, background `#10B981`, border-radius 50% (Strictly static; NO pulsing animations)
  - Version Tag: `JetBrains Mono`, 10px, color `#64748B`, background `rgba(255, 255, 255, 0.03)`, border `1px solid rgba(255, 255, 255, 0.06)`, border-radius 3px

#### HTML / CSS Blueprint
```html
<div class="top-nav-bar">
    <div class="nav-breadcrumbs">
        <span class="nav-brand">◈ CHURNSYNTH</span>
        <span class="nav-slash">/</span>
        <span class="nav-segment">RETAIL ANALYTICS</span>
        <span class="nav-slash">/</span>
        <span class="nav-current">SUBSCRIPTION CHURN SIMULATOR</span>
    </div>
    <div class="nav-status-group">
        <span class="status-pill">
            <span class="status-dot"></span>
            MODEL ACTIVE
        </span>
        <span class="version-tag">v2.4-PRO</span>
    </div>
</div>
```

---

### 3.2 High-Density KPI Metric Ribbon

A 4-column metric ribbon delivering critical cohort diagnostics at a glance. Emphasizes comparison against ground truth with strict numeric alignment.

```
┌───────────────────┬───────────────────┬───────────────────┬───────────────────┐
│ COHORT SIZE  [N]  │ SIMULATED CHURN   │ PROJECTED ARPU    │ AVERAGE TENURE    │
│ 1,000             │ 28.4%             │ $54.20            │ 8.4 mo            │
│ [Active Cohort]   │ [-0.8% vs Real]   │ [+$1.10 vs Real]  │ [+0.2 mo vs Real] │
│ Reverse SDE (100) │ Real: 29.2%       │ Real: $53.10      │ Real: 8.2 mo      │
└───────────────────┴───────────────────┴───────────────────┴───────────────────┘
```

#### Technical Specification
- **Container Grid:** `grid-template-columns: repeat(4, 1fr)`, gap `10px`
- **Card Background:** Surface L1 (`#0F131D`)
- **Card Border:** `1px solid rgba(255, 255, 255, 0.07)` (Hover: `rgba(255, 255, 255, 0.12)`)
- **Card Padding:** `12px 14px`
- **Border Radius:** `6px`
- **Header Row:**
  - Label: `JetBrains Mono`, 10px (`0.625rem`), uppercase, letter-spacing `+0.06em`, color `#64748B`
  - Meta Badge: `JetBrains Mono`, 9.5px, color `#94A3B8`, background `rgba(255, 255, 255, 0.04)`, border `1px solid rgba(255, 255, 255, 0.06)`, padding `1px 5px`, radius `2px`
- **Metric Value:**
  - Font: `JetBrains Mono`, 24px (`1.5rem`), weight 600, color `#F8FAFC`
  - Margin: `4px 0 6px 0`
  - Tabular formatting: forced commas, fixed precision (`.1%`, `$.2f`)
- **Footer Row:**
  - Delta Badge: `JetBrains Mono`, 10px, weight 500, padding `1px 6px`, radius `3px`
  - Delta Neutral: Background `rgba(148, 163, 184, 0.08)`, border `rgba(148, 163, 184, 0.15)`, text `#94A3B8`
  - Delta Positive: Background `rgba(16, 185, 129, 0.08)`, border `rgba(16, 185, 129, 0.20)`, text `#10B981`
  - Delta Negative: Background `rgba(244, 63, 94, 0.08)`, border `rgba(244, 63, 94, 0.20)`, text `#F43F5E`
  - Baseline Label: `JetBrains Mono`, 10.5px, color `#64748B`

#### HTML / CSS Blueprint
```html
<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-header">
            <span class="kpi-label">SIMULATED CHURN RATE</span>
            <span class="kpi-tag">TARGET</span>
        </div>
        <div class="kpi-value">28.4%</div>
        <div class="kpi-footer">
            <span class="delta-badge delta-pos">-0.8% vs Real</span>
            <span class="baseline-text">Baseline: 29.2%</span>
        </div>
    </div>
</div>
```

---

### 3.3 Linear-Grade Segmented Tab Navigation

Tab controls designed as a mechanical segmented control bar embedded flush into the layout.

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ [ Data Explorer & Export ]  [ Distribution Fidelity ]  [ Correlation ]  [ ML Utility ]  │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

#### Technical Specification
- **Track Container:**
  - Background: Surface L1 (`#0F131D`)
  - Border: `1px solid rgba(255, 255, 255, 0.07)`
  - Border Radius: `6px`
  - Padding: `3px`
  - Gap: `3px`
- **Tab Item (Default / Inactive):**
  - Height: `32px`
  - Background: Transparent
  - Border: None
  - Color: `#94A3B8`
  - Font: Primary Sans, 12.5px (`0.781rem`), weight 500
  - Padding: `0 14px`
  - Hover: Background `rgba(255, 255, 255, 0.04)`, Color `#F8FAFC`
- **Tab Item (Active / Selected):**
  - Background: Surface L3 (`#1E2638`)
  - Border: `1px solid rgba(255, 255, 255, 0.12)`
  - Border Radius: `4px`
  - Color: `#F8FAFC`
  - Font Weight: 600
  - Box Shadow: `0 1px 2px rgba(0, 0, 0, 0.4)`
- **Elimination:** Zero glowing blue underline bars or fuzzy ambient backdrops (`stTabs [data-baseweb="tab-highlight"] { display: none !important; }`).

---

### 3.4 Compact Parameter Sidebar

A purpose-built side-dock providing real-time generative parameter steering and architectural validation.

```
┌───────────────────────────────────────────────┐
│ GENERATOR PARAMETERS                          │
│ Configure reverse diffusion sampling engine   │
│                                               │
│ Synthetic Cohort Size (N)             [1,000] │
│ ────●─────────────────────────────────────── │
│                                               │
│ Diffusion Temperature                  [0.95] │
│ ────────────────────●─────────────────────── │
│                                               │
│ ┌───────────────────────────────────────────┐ │
│ │ ARCHITECTURE SPECIFICATIONS               │ │
│ │ Model Family             Tabular DDPM     │ │
│ │ Feature Slices           12 Continuous    │ │
│ │ Denoising Steps (T)      100 Steps        │ │
│ │ Compute Device           MPS / CUDA       │ │
│ └───────────────────────────────────────────┘ │
│                                               │
│ [   SAMPLE SYNTHETIC COHORT (REVERSE SDE)   ] │
└───────────────────────────────────────────────┘
```

#### Technical Specification
- **Sidebar Width:** 320px
- **Sidebar Background:** `#0A0D15`
- **Border Right:** `1px solid rgba(255, 255, 255, 0.07)`
- **Section Headers:** `JetBrains Mono`, 10px, uppercase, tracking `+0.08em`, color `#64748B`, margin `16px 0 8px 0`
- **Spec Box (Key-Value):**
  - Background: Surface L1 (`#0F131D`)
  - Border: `1px solid rgba(255, 255, 255, 0.06)`
  - Border Radius: `6px`
  - Padding: `10px 12px`
  - Row Padding: `4px 0`
  - Row Border Bottom: `1px solid rgba(255, 255, 255, 0.03)` (last row none)
  - Key Column: Primary Sans, 11.5px, color `#94A3B8`
  - Value Column: `JetBrains Mono`, 11.5px, weight 500, color `#F8FAFC`
- **Primary Generator CTA Button:**
  - Background: Solid Indigo `#4F46E5` (No rainbow gradient)
  - Border: `1px solid rgba(255, 255, 255, 0.12)`
  - Border Radius: `5px`
  - Font: Primary Sans, 12.5px, weight 600, color `#FFFFFF`
  - Height: `36px`
  - Hover: Background `#4338CA`, border `rgba(255, 255, 255, 0.20)`
  - Active: Background `#3730A3`
  - Shadow: `0 1px 2px rgba(0, 0, 0, 0.3)` (Zero neon glow)

---

### 3.5 Premium Data Table Specification

For synthetic customer inspection, distribution comparisons, and ML performance scorecards.

#### Technical Specification
- **Wrapper:** Surface L1 (`#0F131D`), border `1px solid rgba(255, 255, 255, 0.07)`, border-radius `6px`
- **Header Cells:**
  - Font: `JetBrains Mono`, 10px, weight 600, uppercase, letter-spacing `+0.05em`
  - Background: `#0A0E17`
  - Color: `#94A3B8`
  - Border Bottom: `1px solid rgba(255, 255, 255, 0.08)`
  - Padding: `8px 12px`
- **Body Cells:**
  - Font: `JetBrains Mono` for IDs, values, dates; Primary Sans for categories
  - Size: 12px (`0.75rem`), weight 400
  - Height: `32px` per row
  - Text Color: `#CBD5E1`
  - Hover Row: Background `#161C2A`
  - Border Bottom: `1px solid rgba(255, 255, 255, 0.04)`
- **Column Alignment Rules:**
  - Identifiers (`customer_id`): Left-aligned, monospace, color `#94A3B8`
  - Text Categories (`subscription_tier`): Left-aligned, sans-serif
  - Numerical Values (`monthly_fee`, `tenure_months`): Right-aligned, monospace, tabular
  - Boolean / Flags (`churned`): Center-aligned, colored badge pill (`0` = Emerald tint, `1` = Rose tint)

---

### 3.6 Executive Plotly Visualization Theme

All charts must integrate seamlessly into the matte dark canvas. Charts should appear as if etched directly into the UI rather than sitting in opaque third-party containers.

#### Technical Configuration Rules
1. **Background:** Both `paper_bgcolor` and `plot_bgcolor` MUST be `"rgba(0, 0, 0, 0)"` (100% transparent).
2. **Gridlines:** Subtle thin lines: `rgba(255, 255, 255, 0.05)`.
3. **Zero Lines:** Low-contrast neutral: `rgba(255, 255, 255, 0.10)`.
4. **Axis Fonts:** `JetBrains Mono`, 10.5px, color `#64748B`.
5. **Legend:**
   - Background: `rgba(15, 19, 29, 0.85)`
   - Border: `1px solid rgba(255, 255, 255, 0.08)`
   - Border Radius: `4px`
   - Font: Primary Sans, 11px, color `#CBD5E1`
6. **Color Palette:**
   - Real Data / Ground Truth: Indigo `#6366F1` (Subdued border `#818CF8`)
   - Synthetic Data / DDPM: Emerald `#10B981` (Subdued border `#34D399`)
   - Divergence / Drift: Rose `#F43F5E`
   - Neutral Correlation Zero: `#0F172A`
7. **Tooltips (Hoverlabel):**
   - Background: `#0F131D`
   - Border: `1px solid rgba(255, 255, 255, 0.15)`
   - Font: `JetBrains Mono`, 11px, color `#F8FAFC`

#### Python Implementation Blueprint
```python
def get_executive_layout(title: str = "", subtitle: str = "", height: int = 360, legend: dict = None) -> dict:
    """Returns authoritative Plotly layout matching ChurnSynth design specifications."""
    base_legend = dict(
        bgcolor="rgba(15, 19, 29, 0.85)",
        bordercolor="rgba(255, 255, 255, 0.08)",
        borderwidth=1,
        font=dict(family="Plus Jakarta Sans, -apple-system, sans-serif", size=11, color="#CBD5E1")
    )
    if legend:
        base_legend.update(legend)

    title_html = (
        f"<span style='font-family: Plus Jakarta Sans; font-size: 13.5px; font-weight: 600; color: #F8FAFC;'>{title}</span>"
        f"{f'<br><span style=\"font-family: Plus Jakarta Sans; font-size: 11px; font-weight: 400; color: #64748B;\">{subtitle}</span>' if subtitle else ''}"
    )

    return dict(
        title=dict(
            text=title_html,
            x=0.01,
            y=0.96,
            xanchor="left",
            yanchor="top"
        ),
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        font=dict(family="Plus Jakarta Sans, sans-serif", size=11, color="#94A3B8"),
        height=height,
        margin=dict(l=45, r=25, t=52 if subtitle else 38, b=42),
        hoverlabel=dict(
            bgcolor="#0F131D",
            bordercolor="rgba(255, 255, 255, 0.15)",
            font=dict(family="JetBrains Mono, monospace", size=11, color="#F8FAFC")
        ),
        xaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.05)",
            zerolinecolor="rgba(255, 255, 255, 0.10)",
            tickfont=dict(family="JetBrains Mono, monospace", size=10.5, color="#64748B"),
            titlefont=dict(family="Plus Jakarta Sans, sans-serif", size=11, color="#94A3B8"),
            linecolor="rgba(255, 255, 255, 0.07)",
            linewidth=1
        ),
        yaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.05)",
            zerolinecolor="rgba(255, 255, 255, 0.10)",
            tickfont=dict(family="JetBrains Mono, monospace", size=10.5, color="#64748B"),
            titlefont=dict(family="Plus Jakarta Sans, sans-serif", size=11, color="#94A3B8"),
            linecolor="rgba(255, 255, 255, 0.07)",
            linewidth=1
        ),
        legend=base_legend
    )
```

---

### 3.7 Streamlined Filter & Parameter Controls

Filter components must fit into tight analytical rows without giant pills or bulky wrappers.

#### Technical Specification
- **Multi-Select & Selectbox Track:**
  - Background: Surface L1 (`#0F131D`)
  - Border: `1px solid rgba(255, 255, 255, 0.07)`
  - Border Radius: `4px`
  - Height: `32px`
- **Selected Tag Inside Multiselect:**
  - Background: Surface L3 (`#1E2638`)
  - Border: `1px solid rgba(255, 255, 255, 0.10)`
  - Border Radius: `3px`
  - Font: `JetBrains Mono`, 11px, color `#CBD5E1`
  - Padding: `1px 6px`
- **Live Summary Micro-Bar:**
  - Placed directly above data tables to show active query filters and resulting record count, churn count, and projected revenue.
  - Background: `#0B0E17`, border: `1px solid rgba(255, 255, 255, 0.06)`, border-radius: `4px`
  - Font: `JetBrains Mono`, 11px

---

## 4. Production Master CSS Implementation

The following complete CSS block replaces the current ad-hoc styles in `app.py`. It is engineered for 100% compatibility with Streamlit's shadow DOM and BaseWeb components.

```css
/* ==========================================================================
   CHURNSYNTH MASTER DESIGN SYSTEM (LINEAR x VERCEL x BLOOMBERG)
   ========================================================================== */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* --- 1. Global Canvas & CSS Variables --- */
:root {
    --cs-canvas: #080A0F;
    --cs-surface-sidebar: #0A0D15;
    --cs-surface-l1: #0F131D;
    --cs-surface-l2: #161C2A;
    --cs-surface-l3: #1E2638;
    
    --cs-border-subtle: rgba(255, 255, 255, 0.05);
    --cs-border-default: rgba(255, 255, 255, 0.07);
    --cs-border-hover: rgba(255, 255, 255, 0.12);
    --cs-border-active: rgba(255, 255, 255, 0.20);
    
    --cs-text-primary: #F8FAFC;
    --cs-text-secondary: #94A3B8;
    --cs-text-tertiary: #64748B;
    --cs-text-disabled: #334155;
    
    --cs-brand: #4F46E5;
    --cs-brand-hover: #4338CA;
    --cs-emerald: #10B981;
    --cs-rose: #F43F5E;
    --cs-amber: #F59E0B;
}

/* Global Reset */
html, body, [class*="css"], .stApp {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    background-color: var(--cs-canvas) !important;
    color: var(--cs-text-primary) !important;
    -webkit-font-smoothing: antialiased !important;
    -moz-osx-font-smoothing: grayscale !important;
}

/* Eliminate glowing background gradients */
.stApp {
    background: var(--cs-canvas) !important;
}

header[data-testid="stHeader"] {
    background: transparent !important;
}

/* --- 2. Precision Scrollbars --- */
::-webkit-scrollbar {
    width: 5px;
    height: 5px;
}
::-webkit-scrollbar-track {
    background: var(--cs-canvas);
}
::-webkit-scrollbar-thumb {
    background: var(--cs-border-default);
    border-radius: 2px;
}
::-webkit-scrollbar-thumb:hover {
    background: var(--cs-border-hover);
}

/* --- 3. Top Navigation Bar --- */
.top-nav-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 38px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--cs-border-default);
    margin-bottom: 14px;
}

.nav-breadcrumbs {
    display: flex;
    align-items: center;
    gap: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.70rem;
    letter-spacing: 0.02em;
}

.nav-brand { color: #818CF8; font-weight: 600; }
.nav-slash { color: var(--cs-text-disabled); }
.nav-segment { color: var(--cs-text-tertiary); font-weight: 500; }
.nav-current { color: var(--cs-text-primary); font-weight: 600; }

.nav-status-group {
    display: flex;
    align-items: center;
    gap: 6px;
}

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: var(--cs-surface-l1);
    border: 1px solid var(--cs-border-default);
    color: var(--cs-text-secondary);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    font-weight: 500;
    padding: 2px 7px;
    border-radius: 3px;
}

.status-dot {
    width: 5px;
    height: 5px;
    background-color: var(--cs-emerald);
    border-radius: 50%;
}

.version-tag {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid var(--cs-border-subtle);
    color: var(--cs-text-tertiary);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    padding: 2px 5px;
    border-radius: 3px;
}

/* --- 4. Hero Section --- */
.hero-title {
    font-size: 1.35rem;
    font-weight: 600;
    letter-spacing: -0.025em;
    line-height: 1.2;
    color: var(--cs-text-primary);
    margin: 0 0 3px 0;
}

.hero-subtitle {
    font-size: 0.81rem;
    color: var(--cs-text-secondary);
    line-height: 1.45;
    max-width: 900px;
    margin-bottom: 10px;
    font-weight: 400;
}

.enterprise-tag-list {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-bottom: 14px;
}

.tag-item {
    display: inline-flex;
    align-items: center;
    background: var(--cs-surface-l1);
    border: 1px solid var(--cs-border-default);
    color: #CBD5E1;
    padding: 2px 7px;
    border-radius: 3px;
    font-size: 0.68rem;
    letter-spacing: 0.01em;
}

.tag-label {
    color: var(--cs-text-tertiary);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-right: 5px;
}

/* --- 5. High-Density KPI Metric Grid --- */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin-bottom: 16px;
}

@media (max-width: 900px) {
    .kpi-grid { grid-template-columns: repeat(2, 1fr); }
}

.kpi-card {
    background: var(--cs-surface-l1);
    border: 1px solid var(--cs-border-default);
    border-radius: 6px;
    padding: 10px 12px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.25);
    transition: border-color 0.12s ease;
}

.kpi-card:hover {
    border-color: var(--cs-border-hover);
}

.kpi-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 4px;
}

.kpi-label {
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--cs-text-tertiary);
    font-family: 'JetBrains Mono', monospace;
}

.kpi-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    color: var(--cs-text-tertiary);
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid var(--cs-border-subtle);
    padding: 1px 4px;
    border-radius: 2px;
}

.kpi-value {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.45rem;
    font-weight: 600;
    color: var(--cs-text-primary);
    letter-spacing: -0.02em;
    line-height: 1.15;
    margin-bottom: 4px;
    font-variant-numeric: tabular-nums;
}

.kpi-footer {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.68rem;
}

.delta-badge {
    display: inline-flex;
    align-items: center;
    gap: 3px;
    padding: 1px 5px;
    border-radius: 2px;
    font-size: 0.64rem;
    font-weight: 500;
    font-family: 'JetBrains Mono', monospace;
    font-variant-numeric: tabular-nums;
}

.delta-neutral {
    background: rgba(148, 163, 184, 0.08);
    color: var(--cs-text-secondary);
    border: 1px solid rgba(148, 163, 184, 0.15);
}

.delta-pos {
    background: rgba(16, 185, 129, 0.08);
    color: var(--cs-emerald);
    border: 1px solid rgba(16, 185, 129, 0.20);
}

.delta-neg {
    background: rgba(244, 63, 94, 0.08);
    color: var(--cs-rose);
    border: 1px solid rgba(244, 63, 94, 0.20);
}

.baseline-text {
    color: var(--cs-text-tertiary);
    font-size: 0.65rem;
    font-family: 'JetBrains Mono', monospace;
}

/* --- 6. Linear-Grade Segmented Tabs --- */
.stTabs [data-baseweb="tab-list"] {
    background-color: var(--cs-surface-l1) !important;
    border: 1px solid var(--cs-border-default) !important;
    border-radius: 6px !important;
    padding: 3px !important;
    gap: 3px !important;
    margin-bottom: 14px !important;
}

.stTabs [data-baseweb="tab"] {
    height: 32px !important;
    border-radius: 4px !important;
    color: var(--cs-text-secondary) !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    padding: 0 14px !important;
    border: none !important;
    background-color: transparent !important;
    transition: all 0.12s ease !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: var(--cs-text-primary) !important;
    background-color: rgba(255, 255, 255, 0.04) !important;
}

.stTabs [aria-selected="true"] {
    background: var(--cs-surface-l3) !important;
    border: 1px solid var(--cs-border-hover) !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.35) !important;
}

.stTabs [data-baseweb="tab-highlight"] {
    display: none !important;
}

/* --- 7. Sidebar Parameter Panel --- */
section[data-testid="stSidebar"] {
    background-color: var(--cs-surface-sidebar) !important;
    border-right: 1px solid var(--cs-border-default) !important;
}

.sidebar-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--cs-text-primary);
    letter-spacing: -0.015em;
    margin-bottom: 2px;
}

.sidebar-subtitle {
    font-size: 0.74rem;
    color: var(--cs-text-tertiary);
    margin-bottom: 12px;
    line-height: 1.4;
}

.sidebar-section-title {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--cs-text-tertiary);
    margin-top: 14px;
    margin-bottom: 8px;
    font-family: 'JetBrains Mono', monospace;
}

.sidebar-spec-box {
    background: var(--cs-surface-l1);
    border: 1px solid var(--cs-border-subtle);
    border-radius: 6px;
    padding: 8px 10px;
    margin-bottom: 12px;
}

.sidebar-spec-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.74rem;
    padding: 4px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.03);
}

.sidebar-spec-row:last-child {
    border-bottom: none;
}

.spec-key { color: var(--cs-text-secondary); }
.spec-val {
    color: var(--cs-text-primary);
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
}

/* Primary Action Button (No rainbow glow) */
button[data-testid="baseButton-primary"] {
    background: var(--cs-brand) !important;
    border: 1px solid var(--cs-border-hover) !important;
    border-radius: 5px !important;
    font-weight: 600 !important;
    font-size: 0.78rem !important;
    padding: 6px 14px !important;
    color: #FFFFFF !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.3) !important;
    transition: background-color 0.12s ease, border-color 0.12s ease !important;
}

button[data-testid="baseButton-primary"]:hover {
    background: var(--cs-brand-hover) !important;
    border-color: var(--cs-border-active) !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.4) !important;
}

/* --- 8. Dataframe & Table Styling --- */
.stDataFrame {
    border: 1px solid var(--cs-border-default) !important;
    border-radius: 6px !important;
    overflow: hidden !important;
}

/* --- 9. Clean Callout Boxes --- */
.callout-box {
    background: var(--cs-surface-l1);
    border: 1px solid var(--cs-border-default);
    border-radius: 5px;
    padding: 10px 12px;
    margin-bottom: 12px;
    display: flex;
    align-items: flex-start;
    gap: 10px;
}

.callout-icon {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    font-weight: 600;
    line-height: 1.4;
    color: #818CF8;
}

.callout-text {
    font-size: 0.78rem;
    color: #CBD5E1;
    line-height: 1.45;
}
```

---

## 5. Implementation Roadmap for Frontend Engineers

To immediately execute this design upgrade on `app.py`:

1. **Replace CSS Injection Block (Lines 27–493):**
   - Strip out all `radial-gradient` backgrounds, pulsing drop shadows, 14px border radii, and rainbow gradient button styles.
   - Inject the Master CSS block from Section 4 above.

2. **Update Plotly Theme Generator (`get_executive_layout`):**
   - Replace the existing function with the specification in Section 3.6.
   - Enforce transparent plot/paper backgrounds, discrete 10.5px `JetBrains Mono` tick markers, and dark slate tooltip styling.

3. **Refactor KPI Metric Cards:**
   - Replace `.glass-card` classes with `.kpi-card`, `.kpi-grid`, and `.delta-badge` classes conforming to the 10px micro-label / 24px mono value layout.

4. **Sanitize Emoji Usage:**
   - Replace cartoon icons (e.g. `ℹ️`) with clean Unicode symbols (`ℹ`, `◈`, `[OK]`, `Δ`).

5. **Tune Data Table Columns:**
   - Configure Streamlit's `st.dataframe` with column configurations (`st.column_config.NumberColumn`, `st.column_config.ProgressColumn`) to ensure numerical columns show currency symbols and tabular decimal alignment.

---

*Authored by Principal UI/UX Designer & Design Systems Architect.*  
*Specification stored at: `/Users/vegirajusrujayvarma/.gemini/antigravity/scratch/churnsynth_ai/DESIGN_SYSTEM.md`*
