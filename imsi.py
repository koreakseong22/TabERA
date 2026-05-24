import openml
import numpy as np

ds = openml.datasets.get_dataset(43454, download_data=True, download_qualities=False, download_features_meta_data=False)
df, y, _, _ = ds.get_data(dataset_format="dataframe")

print("columns:", df.columns.tolist())
print("\ndf shape:", df.shape)
print("\n마지막 5개 컬럼 value_counts:")
for col in df.columns[-5:]:
    print(f"\n{col}:")
    print(df[col].value_counts().head(10))