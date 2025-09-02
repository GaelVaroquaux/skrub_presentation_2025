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
###################################################################
# Split the data into training and testing sets
from sklearn.model_selection import train_test_split    
df_train, df_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)

# %%
# Import sklearn and rock and roll
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
model.fit(df_train, y_train)




# %%
# Joker! Let's use skrub's TableReport to look at the data
from skrub import TableReport
TableReport(df_train)

# %%
# We need to turn this mess to numbers

# First, the date columns
df_train['date_first_hired'] = pd.to_datetime(df_train['date_first_hired'])
df_train['date_first_hired'] = df_train['date_first_hired'].dt.year

# Let's drop the columns with too many unique values
df_train = df_train.drop(['division', 'employee_position_title'], axis=1)

# And let's dummy encode (one-hot encode) the rest
X_train = pd.get_dummies(df_train)

# %%
# Now we can train the model
model.fit(X_train, y_train)

# Hurrah!!

# %%
# But we need to evaluate the model
# And we need to apply the same preparation to the test data...
df_test['date_first_hired'] = pd.to_datetime(df_test['date_first_hired'])
df_test['date_first_hired'] = df_test['date_first_hired'].dt.year
df_test = df_test.drop(['division', 'employee_position_title'], axis=1)
X_test = pd.get_dummies(df_test)

# %%
# Now we can predict, to evaluate the model
y_pred = model.predict(X_test)




# %%
# What's happening??
# Feature names now missing?





# The problem is that the one-hot encoding changed the number of columns
# So the model is now confused about which columns are which

# The right way to do this is to use sklearn's OneHotEncoder
# We need to start again on the training data
from sklearn.preprocessing import OneHotEncoder
# (nitty-gritty details, those options are actually important)
onehot = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
onehot.set_output(transform='pandas')
# We need to one-hot encode only the categorical columns
X_train = pd.concat([onehot.fit_transform(df_train.select_dtypes(include=['object'])),
                     df_train.select_dtypes(exclude=['object'])], axis=1)   

# %%
# We can fit the machine learning model again
model.fit(X_train, y_train)

# %%
# And now we can transform the test data
# This time, we use onehot's transform method, not fit_transform
X_test = pd.concat([onehot.transform(df_test.select_dtypes(include=['object'])),
                     df_test.select_dtypes(exclude=['object'])], axis=1)   

# %%
# And we can evaluate the model!!
y_pred = model.predict(X_test)
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)



# Pff, lot's of wrangling, but we got there

# Ideally, we would do cross-validation, possibly hyperparameter tuning:
# Rerun the train-test split, one-hot encoding, and model fitting multiple times


# %%
# For this, we need to use sklearn's Pipeline, ColumnTransformer...
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import FunctionTransformer
# Define the preprocessing steps
preprocessor = make_column_transformer(
        (OneHotEncoder(sparse_output=False, handle_unknown='ignore'),
         ['gender', 'department', 'department_name', 'assignment_category']),
        # The columns to drop
        ('drop', ['division', 'employee_position_title']),
        # The date column, with the FunctionTransformer
        (FunctionTransformer(lambda this_df: 
            this_df.apply(lambda col: pd.to_datetime(col, errors='coerce').dt.year)),
         ['date_first_hired']),
         remainder='passthrough',
)

# Create a pipeline that includes the preprocessor and the model
pipeline = make_pipeline(preprocessor, RandomForestRegressor())

# We need to redo the train-test split (we have overriden df_train and df_test)
df_train, df_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)


# Fit the pipeline on the training data
pipeline.fit(df_train, y_train)

# Evaluate the model using the pipeline
# The really important aspect is that the model now applies to the raw data
# (no more overwriting the dataframes)
y_pred = pipeline.predict(df_test)
r2_score(y_test, y_pred)

# Actually, with modern versions of sklearn, we do not use to one-hot encode for RandomForest

# %%
# Now let's do the cross-validation
from sklearn.model_selection import cross_val_score
cross_val_score(pipeline, df, y, scoring='r2', n_jobs=-1)
# %%
# And we can also do hyperparameter tuning
from sklearn.model_selection import GridSearchCV
param_grid = {'randomforestregressor__max_depth': [None, 10, 20]}
grid = GridSearchCV(pipeline, param_grid, scoring='r2', n_jobs=-1)
cross_val_score(grid, df, y, scoring='r2')


# Phew, build the data preparation pipeline was a lot of work


# %%
###################################################################
# Let's do it with skrub
from skrub import TableVectorizer
tab_model = make_pipeline(TableVectorizer(), RandomForestRegressor())
# What's in this model?
tab_model
# %%
# The model can be readily applied to dataframes
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
