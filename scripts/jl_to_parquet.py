import sys, json, pandas as pd, hashlib
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf8")]
df = pd.DataFrame(rows).drop_duplicates("text")
df.to_parquet("snapshot.parquet", index=False)
print("Rows:", len(df))

out = "services/text/data/snapshot.parquet"
df.to_parquet(out, index=False)
print("Wrote", out)