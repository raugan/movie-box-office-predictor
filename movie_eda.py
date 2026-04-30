# ============================================================
# MOVIE BOX OFFICE PREDICTOR
# Phase 1: Data Loading & Exploratory Data Analysis
# ============================================================
# Dataset: TMDB 5000 Movies
# Download from: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata
# Files needed: tmdb_5000_movies.csv, tmdb_5000_credits.csv
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import ast
import warnings

warnings.filterwarnings("ignore")

# ── Plot style ────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d0d0d",
    "axes.facecolor":   "#141414",
    "axes.edgecolor":   "#333333",
    "axes.labelcolor":  "#cccccc",
    "xtick.color":      "#888888",
    "ytick.color":      "#888888",
    "text.color":       "#eeeeee",
    "grid.color":       "#222222",
    "grid.linestyle":   "--",
    "font.family":      "monospace",
    "figure.dpi":       130,
})
ACCENT = "#e6b800"   # gold — cinema feel
DIM    = "#555555"

# ============================================================
# STEP 1 — LOAD & MERGE DATASETS
# ============================================================

movies  = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

# The credits file uses 'movie_id'; rename to merge on 'id'
credits.rename(columns={"movie_id": "id"}, inplace=True)
df = movies.merge(credits, on="id")

print(f"Dataset shape after merge: {df.shape}")
print(f"Columns: {list(df.columns)}\n")

# ============================================================
# STEP 2 — PARSE JSON COLUMNS
# ============================================================
# Several columns are stored as stringified JSON lists/dicts.
# We need to extract useful fields from them.

def safe_parse(val):
    """Safely parse a stringified list/dict."""
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return []

def extract_names(val, key="name", limit=None):
    """Extract a list of 'name' values from a JSON column."""
    parsed = safe_parse(val)
    names = [item[key] for item in parsed if key in item]
    return names[:limit] if limit else names

# Genres → list of genre names
df["genre_list"] = df["genres"].apply(extract_names)
df["primary_genre"] = df["genre_list"].apply(
    lambda x: x[0] if x else "Unknown"
)

# Keywords
df["keyword_list"] = df["keywords"].apply(extract_names)

# Production companies
df["company_list"] = df["production_companies"].apply(extract_names)

# Cast — top 3 billed actors
df["cast_list"] = df["cast"].apply(lambda x: extract_names(x, limit=3))

# Director from crew
def get_director(crew_str):
    crew = safe_parse(crew_str)
    for member in crew:
        if member.get("job") == "Director":
            return member.get("name", "Unknown")
    return "Unknown"

df["director"] = df["crew"].apply(get_director)

print("Parsed columns: genre_list, primary_genre, keyword_list,")
print("                cast_list, director, company_list\n")

# ============================================================
# STEP 3 — CLEAN NUMERIC COLUMNS
# ============================================================

# Zero budget/revenue means data is missing — replace with NaN
df["budget"]  = df["budget"].replace(0, np.nan)
df["revenue"] = df["revenue"].replace(0, np.nan)

# Release date → datetime, then extract month & year
df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
df["release_year"]  = df["release_date"].dt.year
df["release_month"] = df["release_date"].dt.month

# Runtime: keep rows with sensible values (60–300 min)
df.loc[(df["runtime"] < 60) | (df["runtime"] > 300), "runtime"] = np.nan

print("Cleaned: budget, revenue, release_date, runtime\n")

# ============================================================
# STEP 4 — DEFINE THE TARGET VARIABLE
# ============================================================
# A movie is a "HIT" if revenue >= 2× budget (covers prints,
# advertising, and production cost at minimum).
# This is a binary classification problem.

df_model = df.dropna(subset=["budget", "revenue"]).copy()
df_model["roi"] = df_model["revenue"] / df_model["budget"]
df_model["hit"] = (df_model["roi"] >= 2.0).astype(int)

hit_rate = df_model["hit"].mean() * 100
print(f"Movies with budget + revenue data: {len(df_model)}")
print(f"Hit rate (ROI >= 2×): {hit_rate:.1f}%")
print(f"Class balance → Hit: {df_model['hit'].sum()} | "
      f"Flop: {(df_model['hit']==0).sum()}\n")

# ============================================================
# STEP 5 — MISSING VALUE SUMMARY
# ============================================================

missing = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
missing = missing[missing > 0]
print("Missing values (%):\n")
print(missing.to_string())
print()

# ============================================================
# STEP 6 — EDA VISUALISATIONS
# ============================================================

fig = plt.figure(figsize=(18, 20))
fig.suptitle("MOVIE BOX OFFICE PREDICTOR — EDA", 
             fontsize=18, fontweight="bold", color=ACCENT,
             y=0.98)

# ── 6a. Hit vs Flop count ─────────────────────────────────────
ax1 = fig.add_subplot(3, 3, 1)
counts = df_model["hit"].value_counts()
bars = ax1.bar(["Flop", "Hit"], counts.values,
               color=[DIM, ACCENT], width=0.5, edgecolor="none")
for bar, val in zip(bars, counts.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
             str(val), ha="center", va="bottom", fontsize=11, color="#eeeeee")
ax1.set_title("Target distribution", color=ACCENT, pad=10)
ax1.set_ylabel("Count")
ax1.grid(axis="x", alpha=0)

# ── 6b. Budget distribution (log scale) ──────────────────────
ax2 = fig.add_subplot(3, 3, 2)
ax2.hist(np.log10(df_model["budget"].dropna()), bins=40,
         color=ACCENT, alpha=0.85, edgecolor="none")
ax2.set_title("Budget distribution (log₁₀ $)", color=ACCENT, pad=10)
ax2.set_xlabel("log₁₀(budget)")
ax2.set_ylabel("Count")
ax2.grid(axis="y")

# ── 6c. Revenue distribution (log scale) ─────────────────────
ax3 = fig.add_subplot(3, 3, 3)
ax3.hist(np.log10(df_model["revenue"].dropna()), bins=40,
         color="#4fc3f7", alpha=0.85, edgecolor="none")
ax3.set_title("Revenue distribution (log₁₀ $)", color="#4fc3f7", pad=10)
ax3.set_xlabel("log₁₀(revenue)")
ax3.set_ylabel("Count")
ax3.grid(axis="y")

# ── 6d. Hit rate by primary genre ────────────────────────────
ax4 = fig.add_subplot(3, 3, 4)
genre_hit = (df_model.groupby("primary_genre")["hit"]
             .agg(["mean", "count"])
             .rename(columns={"mean": "hit_rate", "count": "n"}))
genre_hit = genre_hit[genre_hit["n"] >= 15].sort_values("hit_rate", ascending=True)
colors = [ACCENT if v >= 0.5 else DIM for v in genre_hit["hit_rate"]]
ax4.barh(genre_hit.index, genre_hit["hit_rate"] * 100,
         color=colors, edgecolor="none")
ax4.axvline(50, color="#555", linewidth=1, linestyle="--")
ax4.set_title("Hit rate by genre (%)", color=ACCENT, pad=10)
ax4.set_xlabel("Hit rate (%)")
ax4.grid(axis="x", alpha=0.4)

# ── 6e. Hit rate by release month ────────────────────────────
ax5 = fig.add_subplot(3, 3, 5)
month_names = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]
month_hit = df_model.groupby("release_month")["hit"].mean() * 100
bar_colors = [ACCENT if v >= 50 else DIM for v in month_hit.values]
ax5.bar(month_names, month_hit.values, color=bar_colors, edgecolor="none")
ax5.axhline(50, color="#555", linewidth=1, linestyle="--")
ax5.set_title("Hit rate by release month (%)", color=ACCENT, pad=10)
ax5.set_ylabel("Hit rate (%)")
ax5.tick_params(axis="x", labelsize=8)
ax5.grid(axis="y", alpha=0.4)

# ── 6f. Budget vs Revenue scatter (log-log) ──────────────────
ax6 = fig.add_subplot(3, 3, 6)
hits  = df_model[df_model["hit"] == 1]
flops = df_model[df_model["hit"] == 0]
ax6.scatter(np.log10(flops["budget"]), np.log10(flops["revenue"]),
            alpha=0.3, s=10, color=DIM, label="Flop")
ax6.scatter(np.log10(hits["budget"]), np.log10(hits["revenue"]),
            alpha=0.4, s=10, color=ACCENT, label="Hit")
# 2× budget line
x_vals = np.linspace(4, 9, 100)
ax6.plot(x_vals, x_vals + np.log10(2), color="#e53935",
         linewidth=1.5, linestyle="--", label="ROI = 2×")
ax6.set_title("Budget vs Revenue (log scale)", color=ACCENT, pad=10)
ax6.set_xlabel("log₁₀(budget)")
ax6.set_ylabel("log₁₀(revenue)")
ax6.legend(fontsize=8, framealpha=0.2)

# ── 6g. Runtime distribution by hit/flop ─────────────────────
ax7 = fig.add_subplot(3, 3, 7)
ax7.hist(df_model[df_model["hit"]==0]["runtime"].dropna(),
         bins=30, alpha=0.6, color=DIM, label="Flop", edgecolor="none")
ax7.hist(df_model[df_model["hit"]==1]["runtime"].dropna(),
         bins=30, alpha=0.7, color=ACCENT, label="Hit", edgecolor="none")
ax7.set_title("Runtime distribution", color=ACCENT, pad=10)
ax7.set_xlabel("Runtime (minutes)")
ax7.set_ylabel("Count")
ax7.legend(fontsize=9, framealpha=0.2)
ax7.grid(axis="y", alpha=0.4)

# ── 6h. Top 10 most prolific directors (hit rate) ────────────
ax8 = fig.add_subplot(3, 3, 8)
dir_stats = (df_model.groupby("director")["hit"]
             .agg(["mean", "count"])
             .rename(columns={"mean": "hit_rate", "count": "n"}))
dir_stats = dir_stats[dir_stats["n"] >= 5].sort_values("hit_rate", ascending=False).head(10)
dir_colors = [ACCENT if v >= 0.6 else DIM for v in dir_stats["hit_rate"]]
ax8.barh(dir_stats.index[::-1], dir_stats["hit_rate"][::-1] * 100,
         color=dir_colors[::-1], edgecolor="none")
ax8.axvline(50, color="#555", linewidth=1, linestyle="--")
ax8.set_title("Top directors — hit rate (%)", color=ACCENT, pad=10)
ax8.set_xlabel("Hit rate (%)")
ax8.tick_params(axis="y", labelsize=8)
ax8.grid(axis="x", alpha=0.4)

# ── 6i. Vote average vs ROI ───────────────────────────────────
ax9 = fig.add_subplot(3, 3, 9)
ax9.scatter(df_model["vote_average"],
            np.log10(df_model["roi"].clip(lower=0.01)),
            alpha=0.25, s=8,
            c=df_model["hit"].map({1: ACCENT, 0: DIM}))
ax9.axhline(np.log10(2), color="#e53935", linewidth=1.2,
            linestyle="--", label="ROI = 2× threshold")
ax9.set_title("Vote average vs ROI", color=ACCENT, pad=10)
ax9.set_xlabel("Audience vote (TMDB)")
ax9.set_ylabel("log₁₀(ROI)")
ax9.legend(fontsize=8, framealpha=0.2)
ax9.grid(alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("eda_dashboard.png", dpi=130, bbox_inches="tight",
            facecolor="#0d0d0d")
plt.show()
print("\nEDA dashboard saved → eda_dashboard.png")

# ============================================================
# STEP 7 — KEY INSIGHTS SUMMARY
# ============================================================

print("\n" + "="*55)
print("  KEY INSIGHTS FROM EDA")
print("="*55)

# Insight 1: Best genres
best_genre = genre_hit["hit_rate"].idxmax()
best_genre_rate = genre_hit["hit_rate"].max() * 100
print(f"\n1. Best genre: '{best_genre}' with {best_genre_rate:.0f}% hit rate")

# Insight 2: Best release month
best_month = int(month_hit.idxmax())
print(f"2. Best release month: {month_names[best_month-1]} "
      f"({month_hit[best_month]:.0f}% hit rate)")
      f"({month_hit[best_month]:.0f}% hit rate)")

# Insight 3: Budget vs hit rate
df_model["budget_tier"] = pd.qcut(df_model["budget"], q=4,
    labels=["Low", "Mid-Low", "Mid-High", "High"])
budget_hit = df_model.groupby("budget_tier")["hit"].mean() * 100
print(f"3. Hit rate by budget tier:")
for tier, rate in budget_hit.items():
    print(f"     {tier:10s}: {rate:.0f}%")

# Insight 4: Runtime sweet spot
hit_runtime   = df_model[df_model["hit"]==1]["runtime"].median()
flop_runtime  = df_model[df_model["hit"]==0]["runtime"].median()
print(f"4. Median runtime → Hits: {hit_runtime:.0f} min | "
      f"Flops: {flop_runtime:.0f} min")

print("\n" + "="*55)
print("  NEXT STEP: Run movie_features.py for feature engineering")
print("="*55 + "\n")
