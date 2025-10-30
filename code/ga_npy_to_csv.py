from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent   # repo 
DATA = ROOT / "data"

pop_npy    = np.load(DATA / "ga_pop.npy",    allow_pickle=True)
scores_npy = np.load(DATA / "ga_scores.npy", allow_pickle=True)

pop_rows = [[1 if x else 0 for x in mask] for mask in pop_npy]
pd.DataFrame(pop_rows).to_csv(DATA / "ga_pop.csv", index=False)
pd.DataFrame({"best_score": scores_npy}).to_csv(DATA / "ga_scores.csv", index=False)

print(f"saved to: {DATA}")
