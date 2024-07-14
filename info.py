import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('neurips_2021_zenodo_0_0_1.csv')
categorical_columns = df.select_dtypes(include=['object']).columns

for col in categorical_columns:
    print(f"Value counts for {col}:")
    print(df[col].value_counts())
    print("\n")

    # # 可视化分类属性的数量分布
    # plt.figure(figsize=(10, 6))
    # sns.countplot(x=col, data=df, order=df[col].value_counts().index)
    # plt.title(f"Distribution of {col}")
    # plt.xticks(rotation=90)
    # plt.show()

# Value counts for sound_type:
# sound_type
# mosquito      6795
# background    1900
# audio          600
# Name: count, dtype: int64


# Value counts for species:
# species
# an arabiensis                 1985
# an gambiae ss                  737
# culex quinquefasciatus         678
# culex pipiens complex          545
# an funestus ss                 381
# an squamosus                   141
# ma uniformis                   131
# an dirus                       129
# an harrisoni                   124
# ae aegypti                     123
# an maculatus                   117
# an funestus sl                 104
# an coustani                     92
# an maculipalpis                 79
# ae albopictus                   79
# ma africanus                    78
# an quadriannulatus              70
# an albimanus                    55
# an freeborni                    52
# an stephensi                    45
# an gambiae sl                   42
# culex tarsalis                  39
# an gambiae                      39
# an minimus                      20
# an funestus                     18
# culex tigripes                  18
# an coluzzii                     14
# an farauti                      13
# an atroparvus                   13
# an sinensis                     11
# an barbirostris                 10
# an ziemanni                      9
# toxorhynchites brevipalpis       4
# an merus                         4
# an leesoni                       3
# an pharoensis                    3
# coquillettidia sp                2
# an rivulorum                     1
# Name: count, dtype: int64
