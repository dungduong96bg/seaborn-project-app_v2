import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb

model = lgb.Booster(model_file='/kaggle/input/tdc-credit-scoring/final_model.txt')

model.predict(x_dfs[i])
