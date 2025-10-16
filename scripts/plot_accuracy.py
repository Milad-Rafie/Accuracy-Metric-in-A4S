import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

csv_path = Path("tests/data/measures/accuracy.csv")
df = pd.read_csv(csv_path)

# Be resilient to column names
time_col = "time" if "time" in df.columns else None
score_col = "score" if "score" in df.columns else ("value" if "value" in df.columns else df.columns[-1])

# If a time column exists, parse/sort by time
if time_col:
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.sort_values(time_col)

# Plot
plt.figure()
if time_col:
    plt.plot(df[time_col], df[score_col], marker="o")
    plt.xlabel("Time")
else:
    plt.plot(range(len(df)), df[score_col], marker="o")
    plt.xlabel("Evaluation index")

plt.title("Accuracy over evaluations")
plt.ylabel("Accuracy")
plt.grid(True)
plt.tight_layout()

# (optional) also save to file so you can include it in your report
out_png = Path("tests/data/measures/accuracy_plot.png")
plt.savefig(out_png, dpi=160)
print(f"Saved plot -> {out_png.resolve()}")

plt.show()