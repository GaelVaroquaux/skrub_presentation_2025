# DataOps
####################################
#
# Actual data preparation involves multiple steps, across multiple tables
# "pipeline" API is tedious
#
# WIP: https://github.com/skrub-data/skrub/pull/1233

# %%
# Import a more complex dataset, with multiple tables
import skrub
from skrub.datasets import fetch_credit_fraud

dataset = fetch_credit_fraud()
# A first table, linking baskets to fraud
skrub.TableReport(dataset.baskets)
# %%
# A second table, which gives the products in each basket
skrub.TableReport(dataset.products)
# %%
# We need to 1) group the products by basket, 2) join the two tables
#
# An example of basket looks like this
next(iter(dataset.products.groupby('basket_ID')))[1]

# %%
# A groupby calls for an aggregation. How to aggregate the items, models: strings?
# We'll vectorize the table
vectorizer = skrub.TableVectorizer(high_cardinality=skrub.StringEncoder(),
                                   n_jobs=-1)
vectorized_products = vectorizer.fit_transform(dataset.products)

# %%
# We can now aggregate the products and join the tables: pandas operations
aggregated_products = (
   vectorized_products.groupby("basket_ID").agg("mean").reset_index()
)
aggregated_products
# %%
baskets = dataset.baskets.merge(
    aggregated_products, left_on="ID", right_on="basket_ID"
).drop(columns=["ID", "basket_ID"])
baskets

# %%
# And we can now train a model
y = baskets['fraud_flag']
X = baskets.drop('fraud_flag', axis=1)
from sklearn.ensemble import ExtraTreesClassifier # It's fast
model = ExtraTreesClassifier(n_jobs=-1)
from sklearn.model_selection import cross_val_score
cross_val_score(model, X, y, scoring='roc_auc')


# %%
# But we are back to the same problem as in the first section:
# We're in pandas' land. When comes new data, how to apply the same transformations?
# How to cross-validate, or tune the data-preparation steps?



#######################################################################
# skrub comes with a way to change ever so slightly what we did above
# We define our inputs as "variables"
products = skrub.var("products", dataset.products)
products
# %%
# Now we define our "x" and "y" variables
baskets = skrub.var('baskets', dataset.baskets)
basket_IDs = baskets[["ID"]].skb.mark_as_x()
fraud_flags = baskets["fraud_flag"].skb.mark_as_y()

# %%
# We can now proceed almost as above to prepare the data
# We vectorize the products, with a slightly different API
from skrub import selectors as s
vectorized_products = products.skb.apply(vectorizer, cols=s.all() - "basket_ID")
vectorized_products
# %%
# We aggregate the products
aggregated_products = vectorized_products.groupby("basket_ID").agg("mean").reset_index()
aggregated_products
# %%
# And we join the tables
features = basket_IDs.merge(aggregated_products, left_on="ID", right_on="basket_ID")
features = features.drop(columns=["ID", "basket_ID"])
features
# %%
# And we do the prediction
predictions = features.skb.apply(ExtraTreesClassifier(n_jobs=-1), y=fraud_flags)
predictions

# %%
# What's the big deal? We now have a graph of computations
# We can apply it to new data

# We load the test data
data_test = fetch_credit_fraud(split="test")
y_test = data_test.baskets['fraud_flag']
basket_test = data_test.baskets.drop('fraud_flag', axis=1)

# We can apply a predictor to this new data
predictor = predictions.skb.get_pipeline(fitted=True)
y_pred = predictor.predict({
    'baskets': basket_test,
    'products': data_test.products,
})

# And we can evaluate the predictions
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))



# %%
# We can also tune hyperparameters of our data preparation
# We just need to change a bit the above code
encoder = skrub.StringEncoder(
    vectorizer=skrub.choose_from(["tfidf", "hashing"], name="vectorizer"),
)
vectorizer = skrub.TableVectorizer(high_cardinality=encoder,
                                   n_jobs=2)

# The rest of the code remains the same
vectorized_products = products.skb.apply(vectorizer, cols=s.all() - "basket_ID")
aggregated_products = vectorized_products.groupby("basket_ID").agg("mean").reset_index()

# We redefine our sources, to have a clean start
baskets = skrub.var('baskets', dataset.baskets)
basket_IDs = baskets[["ID"]].skb.mark_as_x()
fraud_flags = baskets["fraud_flag"].skb.mark_as_y()
features = basket_IDs.merge(aggregated_products, left_on="ID", right_on="basket_ID")
features = features.drop(columns=["ID", "basket_ID"])
predictions = features.skb.apply(ExtraTreesClassifier(n_jobs=-1), y=fraud_flags)

search = predictions.skb.get_grid_search(fitted=True, scoring="roc_auc",
                                         verbose=2)
search.get_cv_results_table()

# %%
# skrub gives you all kinds of tools to tune and inspect this pipeline:
# For instance, we can visualize the hyperparameters selection
search.plot_parallel_coord()

# %%
# We can also get a full report of the pipeline
predictions.skb.full_report()
# %%
