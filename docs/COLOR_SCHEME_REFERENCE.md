# Color Scheme Reference for Scientific Plots

## Current Implementation

### plot_spi_space (Grid plots)
- **Dots:** `color='gray'` with `alpha=0.7`
- **Regression line colormap:** `plt.cm.RdYlBu_r` (Red-Yellow-Blue reversed)
  - Maps correlation ρ ∈ [-1, +1] to color
  - Red = +1 (perfect positive correlation)
  - Yellow = 0 (no correlation)
  - Blue = -1 (perfect negative correlation)
  - **Why RdYlBu_r?** Avoids white/clear at zero (your concern), diverging colormap is standard for correlation

### plot_spi_space_individual (Single scatter plots)
- **Dots:** `color='gray'` with `alpha=0.7`
- **Regression line:** `#4ECDC4` (soft teal)
  - Single fixed color, no correlation-based variation
  - Pastel, non-distracting, good contrast with gray dots

---

## Pastel Color Palettes for Scientific Visualization

### Diverging Colormaps (For Correlation Values in [-1, +1])

These are designed for data with a meaningful midpoint (zero correlation):

| Colormap | Range | Best For | Notes |
|----------|-------|----------|-------|
| **RdYlBu_r** ✅ | Red → Yellow → Blue | **Current choice** | No white at zero, classic diverging |
| **RdBu_r** | Red → White → Blue | Correlation matrices | White at zero can be too subtle |
| **coolwarm** | Blue → White → Red | General diverging data | Similar to RdBu_r |
| **PiYG** | Pink → White → Green | Biological data | Softer than RdBu |
| **PRGn** | Purple → White → Green | Geological data | More pastel |
| **RdYlGn_r** | Red → Yellow → Green | Traffic light analogy | Intuitive for good/bad |
| **Spectral_r** | Red → Yellow → Green → Blue | Multi-class data | Rainbow-like |

**Recommendation:** Stick with **RdYlBu_r** - it's the scientific standard for correlation and avoids the white-at-zero problem.

---

### Sequential Colormaps (For One-Directional Data [0, ∞) or [0, 1])

Not applicable here since you're plotting correlations in [-1, +1], but useful for MI, TE, DTW:

| Colormap | Range | Best For |
|----------|-------|----------|
| **viridis** | Purple → Blue → Green → Yellow | Perceptually uniform, colorblind-friendly |
| **plasma** | Purple → Pink → Orange → Yellow | Similar to viridis |
| **cividis** | Blue → Yellow | Colorblind-optimized |
| **Blues** | White → Dark Blue | Single-hue, simple |
| **YlOrRd** | Yellow → Orange → Red | Heatmaps, intensity |

---

### Pastel Single Colors (For Regression Lines)

If you want alternatives to soft teal (`#4ECDC4`):

| Color Name | Hex Code | RGB | Use Case |
|------------|----------|-----|----------|
| **Soft Teal** ✅ | `#4ECDC4` | (78, 205, 196) | **Current choice** - calm, scientific |
| **Soft Coral** | `#FA8072` | (250, 128, 114) | Warm alternative |
| **Soft Purple** | `#B19CD9` | (177, 156, 217) | Elegant, academic |
| **Soft Sky Blue** | `#87CEEB` | (135, 206, 235) | Light, airy |
| **Soft Mint** | `#98D8C8` | (152, 216, 200) | Similar to teal, greener |
| **Soft Lavender** | `#E6E6FA` | (230, 230, 250) | Very subtle, may lack contrast |
| **Soft Peach** | `#FFDAB9` | (255, 218, 185) | Warm, gentle |
| **Soft Slate** | `#708090` | (112, 128, 144) | Neutral, professional |

**Recommendation:** Keep **#4ECDC4 (soft teal)** - it has:
- Good contrast with gray dots
- Not too vibrant (doesn't compete with correlation-colored lines in grid plots)
- Scientifically neutral (doesn't imply positive/negative bias)

---

## Colorblind-Friendly Considerations

~8% of males have some form of color vision deficiency. Best practices:

1. **Avoid Red-Green combinations alone** (most common deficiency)
   - RdYlBu_r is OK because it has 3 colors (red, yellow, blue)
   - RdBu_r is risky (red-blue can look similar to protanopes)

2. **Use colormap + markers/patterns** when possible
   - Your plots already do this: gray dots + colored line

3. **Test with simulators**:
   - [Coblis](https://www.color-blindness.com/coblis-color-blindness-simulator/)
   - [Viz Palette](https://projects.susielu.com/viz-palette)

4. **Safe colormaps** (colorblind-friendly):
   - **viridis**, **plasma**, **cividis** (sequential)
   - **RdYlBu_r** is generally OK for diverging (3 distinct hues)

---

## Violin Plots - Why NOT Recommended

You asked: *"do violin plot-esque lines on plot_spi_space() and plot_spi_space_individual() make sense here?"*

**Answer: NO**, here's why:

### What Violin Plots Show
- **Distribution shapes** (via kernel density estimation)
- Compare distributions across categorical groups
- Example: "Distribution of height in males vs females"

### What Your Plots Show
- **Correlation relationships** between two continuous SPIs
- Each point = one edge (i,j) from M×M matrix
- Goal: Visualize if SPI₁ values correlate with SPI₂ values

### Why Violin Plots Don't Apply
1. **You're not comparing groups** - you have a single scatter cloud per SPI pair
2. **Marginal distributions aren't the focus** - you care about the joint relationship (correlation)
3. **Would clutter the plot** - adding marginal violin plots to 171 individual plots would be overwhelming
4. **Correlation is already captured** - the regression line + ρ value quantify the relationship

### What WOULD Make Sense (if you wanted more)
- **Marginal histograms** (like seaborn's `jointplot`) - but only for individual plots, not the grid
- **Confidence bands** around regression line - but probably overkill for exploratory analysis
- **Hexbin plots** for very dense data - but your M=15 gives only 105 points, sparse enough for scatter

**Bottom line:** Your current setup (scatter + regression + correlation value) is optimal for this use case. Don't add violin plots.

---

## Summary Recommendations

### Keep Current Choices ✅
1. **plot_spi_space colormap:** `RdYlBu_r` (Red-Yellow-Blue for correlation)
2. **plot_spi_space_individual line color:** `#4ECDC4` (soft teal)
3. **Dot color:** Gray with α=0.7 (neutral, doesn't compete with line color)
4. **No violin plots:** Scatter + regression is correct for correlation visualization

### Alternatives (if you want to experiment)
1. **Colormap:** Try `coolwarm` or `PRGn` for different aesthetics
2. **Line color:** Try `#FA8072` (soft coral) or `#B19CD9` (soft purple) for warmer/cooler feel
3. **Dot size:** Current s=12 (grid) and s=15 (individual) are good, but test s=10/s=18 if needed

### Final Thought
The color scheme serves the data, not the other way around. Your current choices are:
- **Scientifically appropriate** (RdYlBu_r for correlation is standard)
- **Aesthetically pleasing** (pastel teal, gray dots, good contrast)
- **Functionally clear** (grid plots show correlation strength via color, individual plots use neutral color)

No changes needed unless you have specific accessibility requirements or personal preference.
