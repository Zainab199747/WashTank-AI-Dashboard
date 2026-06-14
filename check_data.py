import pandas as pd

df = pd.read_csv("master_dataset.csv")

print(df[
    [
        "avg_temperature",
        "avg_pressure",
        "avg_level",
        "avg_oil_outlet",
        "avg_water_outlet",
        "daily_dosage",
        "emulsion_level_m"
    ]
].describe())