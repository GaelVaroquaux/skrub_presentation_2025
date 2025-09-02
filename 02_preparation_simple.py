# The data preparation experience
####################################
# %%
# Load the data
import pandas as pd

df = pd.read_csv('employees_salaries.csv')
df.head()
# %%
# Split out the column to predict
y = df['salary']
df = df.drop('salary', axis=1)

# %%
# Use skrub's TableReport to look at the data
from skrub import TableReport
TableReport(df)


# %%
# Import sklearn, skrub and rock and roll
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from skrub import TableVectorizer

tab_model = make_pipeline(TableVectorizer(), RandomForestRegressor())
# What's in this model?
tab_model
# %%
# The model can be readily applied to dataframes
from sklearn.model_selection import cross_val_score
cross_val_score(tab_model, df, y, scoring='r2', n_jobs=-1)
# %%
# We can actually do something faster
from skrub import tabular_pipeline
tab_model = tabular_pipeline(RandomForestRegressor())
# What's in that model?
tab_model
# %%
cross_val_score(tab_model, df, y, scoring='r2', n_jobs=-1)

# %%
