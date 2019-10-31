---
layout:     post
title:      Pandas Dataframe
subtitle:   Dataframe 迭代
date:       2019-10-25
author:     LZY
header-img: img/whatisnext.jpg
catalog: true
tags:
    - Pandas
    - Dataframe
---

# Dataframe

```python
import pandas as pd

path = 'file.csv'
df = pd.read_csv(path)#DataFrame

```

# Method rows

```python
for row in df.rows:
    print(row['c1'])

```

# Method iterrows

```python
for index, row in df.iterrows():
    print(row['c1'])

```

`iterrows`返回的是副本而不是视图，修改不起作用

改用`DataFrame.apply()`：

```python
new_df = df.apply(lambda x: x * 2)
```

# Method itertuples

```python
for row in df.itertuples():
    print(row['c1'])

```

`itertuples`应该比`iterrows`快

# Method apply

`DataFrame.apply()`

```python
def valuation_formula(x, y):
    return x * y * 0.5

df['price'] = df.apply(lambda row: valuation_formula(row['x'], row['y']), axis=1)
```
# Method iloc

```python
for i in range(0, len(df)):
    print df.iloc[i]['c1'], df.iloc[i]['c2']
```


