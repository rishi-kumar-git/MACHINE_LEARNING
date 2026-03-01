from data import build_dataset
from stuff import *
from sklearn.model_selection import train_test_split

dataset_path = r"C:\Users\Rishi\OneDrive\Desktop\SEM 4\machine learning\lab3\Lab2_Dataset"

X, y = build_dataset(dataset_path)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

feature_index = 0

single_model = train_single_attribute_lr(X_train, feature_index)

train_true, train_pred = predict_single_attribute_lr(single_model, X_train, feature_index)
test_true, test_pred = predict_single_attribute_lr(single_model, X_test, feature_index)

train_metrics = regression_scores(train_true, train_pred)
test_metrics = regression_scores(test_true, test_pred)

print(train_metrics)
print(test_metrics)

multi_model = train_multi_attribute_lr(X_train, feature_index)
multi_true, multi_pred = predict_multi_attribute_lr(multi_model, X_test, feature_index)
multi_metrics = regression_scores(multi_true, multi_pred)

print(multi_metrics)

kmeans_model = perform_kmeans(X_train, 2)
labels, centers = get_cluster_details(kmeans_model)

print(centers)

sil, ch, db = clustering_metrics(X_train, labels)

print(sil)
print(ch)
print(db)

k_values = range(2, 10)
sil_scores, ch_scores, db_scores = evaluate_multiple_k(X_train, k_values)

plot_k_analysis(k_values, sil_scores, ch_scores, db_scores)

k_range = range(2, 20)
distortions = elbow_method(X_train, k_range)

plot_elbow(k_range, distortions)
