# DataOps
####################################
#

# %%
# Import a more complex dataset, with multiple tables
import skrub
from skrub.datasets import fetch_credit_fraud

dataset = fetch_credit_fraud(split="train")
# A first table, linking baskets to fraud
baskets_df = dataset.baskets
skrub.TableReport(baskets_df)
# %%
# A second table, which gives the products in each basket
products_df = dataset.products
skrub.TableReport(products_df)
# %%
# We need to 1) group the products by basket, 2) join the two tables
#
# An example of basket looks like this
next(iter(products_df.groupby('basket_ID')))[1]

# %%
# A pandas pipeline
# 1. Split out features and target
basket_IDs = baskets_df[["ID"]]
fraud_flags = baskets_df["fraud_flag"]

# Extract simplified tables
products_simple = products_df[["basket_ID", "cash_price", "Nbr_of_prod_purchas"]]

# 2 .Aggregate the products and join the tables
aggregated_products = products_simple.groupby("basket_ID").agg("mean")
aggregated_products = aggregated_products.reset_index()
features = basket_IDs.merge(aggregated_products, left_on="ID", right_on="basket_ID")

# 3. Train a model
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier(n_jobs=-1)
model.fit(features, fraud_flags)


# %%
# Great, but how do I apply this to new data?
# I need to re-apply the pandas's transformations


#######################################################################
# skrub comes with a way to change ever so slightly what we did above

# 0. Define our inputs as "variables"
products = skrub.var("products", dataset.products)
products

# 1. Split out features and target
baskets = skrub.var("baskets", dataset.baskets)
basket_IDs = baskets[["ID"]].skb.mark_as_X()
fraud_flags = baskets["fraud_flag"].skb.mark_as_y()

# Extract simplified tables
products_simple = products[["basket_ID", "cash_price", "Nbr_of_prod_purchas"]]
products_simple

# %%
# 2 .Aggregate the products and join the tables
aggregated_products = products_simple.groupby("basket_ID").agg("mean")
aggregated_products = aggregated_products.reset_index()
features = basket_IDs.merge(aggregated_products, left_on="ID", right_on="basket_ID")
features

# %%
# 3. Train a model
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier(n_jobs=-1)
predictions = features.skb.apply(model, y=fraud_flags)
predictions

# %%
# What's the big deal? We now have a graph of computations

# %%
# We can also get a full report of the pipeline
predictions.skb.full_report()

# %%
# We can apply it to new data

# We load the test data
data_test = fetch_credit_fraud(split="test")

# We can apply a predictor to this new data
predictor = predictions.skb.make_learner(fitted=True)
y_pred = predictor.predict({
    'baskets': data_test.baskets,
    'products': data_test.products,
})

# And we can evaluate the predictions
from sklearn.metrics import classification_report
print(classification_report(data_test.baskets['fraud_flag'], y_pred))


# %%
#############################################################
# Tune this pipeline

# We redefine our sources, to have a clean start
# 0. Define our inputs as "variables"
products = skrub.var("products", dataset.products)
# 1. Split out features and target
baskets = skrub.var("baskets", dataset.baskets)
basket_IDs = baskets[["ID"]].skb.mark_as_X()
fraud_flags = baskets["fraud_flag"].skb.mark_as_y()
products_simple = products[["basket_ID", "cash_price", "Nbr_of_prod_purchas"]]
products_simple

# %%
# 2. A tunable Aggregate + join
aggregated_products = products_simple.groupby("basket_ID").agg(
    skrub.choose_from(("mean", "max", "count"))
)

aggregated_products = aggregated_products.reset_index()
features = basket_IDs.merge(aggregated_products, left_on="ID", right_on="basket_ID")

# 3. Train a model
model = ExtraTreesClassifier(n_jobs=-1)
predictions = features.skb.apply(model, y=fraud_flags)
predictions

search = predictions.skb.make_grid_search(fitted=True, scoring="roc_auc",
                                         verbose=2)
search.results_

# %%
# skrub gives you all kinds of tools to tune and inspect this pipeline:
# For instance, we can visualize the hyperparameters selection
search.plot_results()

