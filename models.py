 
import numpy as np
import pandas as pd
import os
from feature_engine.outliers import Winsorizer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2, RFE, SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import pickle
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import silhouette_score

import warnings
warnings.filterwarnings("ignore")

 
df = pd.read_csv("train.csv")
df.head()

 
# df["score"].value_counts()
# display(df.info())
# display(df.size)

 
#data cleaning and standardization
df.isnull().sum()
df.dropna(inplace=True)

df['year'] = pd.DatetimeIndex(df['date']).year
df['month'] = pd.DatetimeIndex(df['date']).month
df['day'] = pd.DatetimeIndex(df['date']).day
df['score'] = df['score'].round().astype(int)
df.dtypes

 
# print("df has missing values:", df.isnull().any().any())

 
# We can re-run the value counts to see the rounded scores' counts
df["score"].value_counts()

 
# Check for duplicates
duplicates = df.duplicated().any()
# print(f"Does the DataFrame have duplicated rows? {duplicates}")

 
# For numeric columns
# display(df.describe().T)

# For Categorical columns
# display(df.describe(include=['object']))

 
cols = list(df.columns)
cols

 
#Get numeric features out into a list and remove unnecessary features
measures_list = [x for x in df.select_dtypes(include=["int64", "float64", "int32"])]
remove_list = ["fips", "score", "year", "month", "day"]

# List comprehension for applying remove_list to our measures_list
measures_list = [i for i in measures_list if i not in remove_list]
# Create measures_df out of df with only the numeric features we want to visualize
measures_df = df[measures_list]

 
num_cols = 3  # Number of histograms per figure
num_rows = (len(measures_list) + num_cols - 1) // num_cols  # Calculate number of rows needed

# Create a figure with subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 15), constrained_layout=True)

# Flatten the axes array for easier iteration
axes = axes.flatten()

for i, col_name in enumerate(measures_list):
    ax = axes[i]
    ax.hist(measures_df[col_name], density=True)
    ax.set_xlabel(col_name)
    ax.set_ylabel('Density')
    ax.set_title(f'Distribution of {col_name}')

# Turn off axes for any unused subplots
for j in range(len(measures_list), len(axes)):
    axes[j].axis('off')

plt.show()

 
plt.figure(figsize=(10,40))
for x in (range(1,19)):
    plt.subplot(19,1,x)
    sns.boxplot(x =  measures_df.columns[x-1], data=measures_df)
    x_name = measures_df.columns[x-1]
    plt.title(f'Distribution of {x_name}')
plt.tight_layout()

 
outlier = Winsorizer(capping_method="gaussian",
                     tail="both",
                    fold=3,
                    variables = measures_list,
                    missing_values="ignore")

outlier.fit(df)

 
# What are our new min and max values?
outlier.right_tail_caps_

 
outlier.left_tail_caps_

 
# Apply the Winsorizer to our main DataFrame
df = outlier.transform(df)

 
# Create a new categorical DataFrame for visualization
categorical_column_list = ['score','year','month','day']
df_categorical = df[['score','year','month','day']]

 
plt.figure(figsize=(10,40))
for col_name in categorical_column_list:
    plt.figure()
    df_categorical[col_name].value_counts().plot(kind = 'bar')
    x_name = col_name
    y_name = 'Density'
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title('Distribution of {x_name}'.format(x_name=x_name))
    plt.tight_layout()

 
# Get .corr and visualize with Seaborn for an easier visual analysis of our features
correlation_matrix = measures_df.corr().round(2)

plt.figure(figsize=(12,8))
sns.heatmap(data=correlation_matrix, annot=True, annot_kws={"size": 10, "weight": "bold", "color": "black"}, cmap=sns.diverging_palette(10, 150, as_cmap=True))
plt.title("Correlation Matrix")
plt.show()

 
# We make sure our column names and values are in a single form. In this case there're no string values
df.columns = df.columns.str.lower().str.replace(" ","_")
df.head()

 
# Drop useless columns | remove fips and date
df = df.drop(columns=["date", "fips"])

# X, y
X = df.drop(["score"], axis=1)
y = df["score"]

# Get shapes of our variables and targets
# print("X.shape: {} \ny.shape: {}".format(X.shape, y.shape))

 
# Split our data for train and test sets for validation, test size is 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

 
# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# print("Training data shape:", X_train_scaled.shape)
# print("Test data shape:", X_test_scaled.shape)

 
# Feature selection using RFE with consistent random_state
dt_selector = DecisionTreeClassifier(random_state=42)
rfe = RFE(estimator=dt_selector, n_features_to_select=10)
X_train_rfe = rfe.fit_transform(X_train_scaled, y_train)
X_test_rfe = rfe.transform(X_test_scaled)

# Get selected feature names
selected_features = X.columns[rfe.support_].tolist()
# print("Selected features:", selected_features)
# print("Number of selected features:", len(selected_features))

 
# Re-split data with consistent random_state=42 (FIXED: was using random_state=0 before)
# Using the selected features for consistent comparison
X_selected = df[selected_features]
X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(
    X_selected, y, test_size=0.2, random_state=42
)

# print("Final training data shape:", X_train_final.shape)
# print("Final test data shape:", X_test_final.shape)

 
# Scale the final selected features
scaler_final = StandardScaler()
X_train_scaled_final = scaler_final.fit_transform(X_train_final)
X_test_scaled_final = scaler_final.transform(X_test_final)

 
# Apply SMOTE consistently to both models
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled_final, y_train_final)

# print("Original training set shape:", X_train_scaled_final.shape)
# print("SMOTE training set shape:", X_train_smote.shape)
# print("Original class distribution:", np.bincount(y_train_final))
# print("SMOTE class distribution:", np.bincount(y_train_smote))

 
# Decision Tree with hyperparameter tuning to prevent overfitting
# Cross-validation to find optimal max_depth
dt_param_grid = {
    'max_depth': [3, 5, 7, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

dt_grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    dt_param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

dt_grid_search.fit(X_train_smote, y_train_smote)
# print("Best Decision Tree parameters:", dt_grid_search.best_params_)
# print("Best cross-validation accuracy:", dt_grid_search.best_score_)

# Train final Decision Tree with best parameters
best_dt = dt_grid_search.best_estimator_
y_train_pred_dt = best_dt.predict(X_train_smote)
y_test_pred_dt = best_dt.predict(X_test_scaled_final)

# Check for overfitting by comparing train vs test accuracy
train_acc_dt = accuracy_score(y_train_smote, y_train_pred_dt)
test_acc_dt = accuracy_score(y_test_final, y_test_pred_dt)
# print(f"Decision Tree - Train Accuracy: {train_acc_dt:.4f}")
# print(f"Decision Tree - Test Accuracy: {test_acc_dt:.4f}")
# print(f"Decision Tree - Overfitting gap: {train_acc_dt - test_acc_dt:.4f}")

 
# KNN hyperparameter tuning
knn_param_grid = {
    'n_neighbors': [1, 3, 5, 7, 9, 11, 15, 21],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

knn_grid_search = GridSearchCV(
    KNeighborsClassifier(),
    knn_param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

knn_grid_search.fit(X_train_smote, y_train_smote)
# print("Best KNN parameters:", knn_grid_search.best_params_)
# print("Best cross-validation accuracy:", knn_grid_search.best_score_)

# Train final KNN with best parameters (COMPLETED: was commented out before)
best_knn = knn_grid_search.best_estimator_
y_train_pred_knn = best_knn.predict(X_train_smote)
y_test_pred_knn = best_knn.predict(X_test_scaled_final)

# Check for overfitting
train_acc_knn = accuracy_score(y_train_smote, y_train_pred_knn)
test_acc_knn = accuracy_score(y_test_final, y_test_pred_knn)
# print(f"KNN - Train Accuracy: {train_acc_knn:.4f}")
# print(f"KNN - Test Accuracy: {test_acc_knn:.4f}")
# print(f"KNN - Overfitting gap: {train_acc_knn - test_acc_knn:.4f}")

 
# Evaluation function using full dataset (FIXED: removed 100k subsampling)
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """Evaluate model performance using full dataset"""
    # Training predictions
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train) if hasattr(model, 'predict_proba') else None
    
    # Test predictions
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    
    # Metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    train_precision = precision_score(y_train, y_train_pred, average='weighted')
    test_precision = precision_score(y_test, y_test_pred, average='weighted')
    train_recall = recall_score(y_train, y_train_pred, average='weighted')
    test_recall = recall_score(y_test, y_test_pred, average='weighted')
    
   # # print(f"\n{model_name} Performance:")
   # # print(f"Training - Accuracy: {train_acc:.4f}, F1: {train_f1:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}")
   # # print(f"Test - Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")
    
    return {
        'model': model_name,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'train_f1': train_f1,
        'test_f1': test_f1,
        'train_precision': train_precision,
        'test_precision': test_precision,
        'train_recall': train_recall,
        'test_recall': test_recall,
        'y_test_pred': y_test_pred,
        'y_test_proba': y_test_proba
    }

 
# Evaluate both models using full dataset
dt_results = evaluate_model(best_dt, X_train_smote, y_train_smote, X_test_scaled_final, y_test_final, "Decision Tree")
knn_results = evaluate_model(best_knn, X_train_smote, y_train_smote, X_test_scaled_final, y_test_final, "KNN")

 
# ROC curves using predict_proba (FIXED: was using hard predictions before)
def plot_multiclass_roc(y_true, y_proba, model_name, n_classes=6):
    """Plot ROC curves for multiclass classification using probabilities"""
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc
    
    # Binarize the output
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    plt.figure(figsize=(10, 8))
    
    # Compute ROC curve and ROC area for each class
    for i in range(n_classes):
        if i < y_proba.shape[1]:  # Check if class exists in predictions
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'Class {i} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Multiclass ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.show()

# Plot ROC curves for both models
if dt_results['y_test_proba'] is not None:
    plot_multiclass_roc(y_test_final, dt_results['y_test_proba'], "Decision Tree")

if knn_results['y_test_proba'] is not None:
    plot_multiclass_roc(y_test_final, knn_results['y_test_proba'], "KNN")

 
# Model comparison summary table (NEW: comprehensive comparison)
def create_comparison_table(dt_results, knn_results):
    """Create a comprehensive comparison table for both models"""
    comparison_data = {
        'Model': ['Decision Tree', 'KNN'],
        'Test Accuracy': [dt_results['test_acc'], knn_results['test_acc']],
        'Test F1 Score': [dt_results['test_f1'], knn_results['test_f1']],
        'Test Precision': [dt_results['test_precision'], knn_results['test_precision']],
        'Test Recall': [dt_results['test_recall'], knn_results['test_recall']],
        'Train Accuracy': [dt_results['train_acc'], knn_results['train_acc']],
        'Train F1 Score': [dt_results['train_f1'], knn_results['train_f1']],
        'Overfitting Gap': [dt_results['train_acc'] - dt_results['test_acc'], 
                           knn_results['train_acc'] - knn_results['test_acc']]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.round(4)
    
   # # print("Model Comparison Summary:")
   # # print("=" * 80)
   # # print(comparison_df.to_string(index=False))
   # # print("=" * 80)
    
    # Find best model for each metric
    best_accuracy = comparison_df.loc[comparison_df['Test Accuracy'].idxmax(), 'Model']
    best_f1 = comparison_df.loc[comparison_df['Test F1 Score'].idxmax(), 'Model']
    least_overfitting = comparison_df.loc[comparison_df['Overfitting Gap'].idxmin(), 'Model']
    
   # # print(f"\nBest performing model by accuracy: {best_accuracy}")
   # # print(f"Best performing model by F1 score: {best_f1}")
   # # print(f"Model with least overfitting: {least_overfitting}")
    
    return comparison_df

# Create and display comparison table
comparison_table = create_comparison_table(dt_results, knn_results)

 
# Detailed classification reports for both models
# print("Decision Tree Classification Report:")
# print("=" * 50)
# print(classification_report(y_test_final, dt_results['y_test_pred']))

# print("\nKNN Classification Report:")
# print("=" * 50)
# print(classification_report(y_test_final, knn_results['y_test_pred']))

 
# print("Train features shape", X_train.shape)
# print("Train target shape", y_train.shape)
# print("Test features shape", X_test.shape)
# print("Test target shape", y_test.shape)

 
# Initialize our Standart Scaler
sc = StandardScaler()

# Use .fit and .transform in the same line with fit_transform on X_train
X_train = sc.fit_transform(X_train)
# only transform X_test
X_test = sc.transform(X_test)
# This process doesn't remember the column names and X_train is now a Numpy array not a DataFrame
X_train

 
X_scaled_sub = X_train[:100000]
y_scaled_sub = y_train[:100000]

classifier = RandomForestClassifier(n_estimators=10) # n is 100 by default we reduced it
classifier.fit(X_scaled_sub, y_scaled_sub)

feature_importance = classifier.feature_importances_

plt.figure(figsize=(10, 6))
plt.bar(X.columns, feature_importance)
# plt.bar(X_train.columns, feature_importance)

plt.xlabel('Features')
plt.ylabel('Importance Scores')
plt.title('Feature Importance Scores')
plt.xticks(rotation=45)
plt.show()

 
# Created a backup copy for ease of use while developing the code
Xi = X.copy()

# Initiate Random Forest Classifier
model = RandomForestClassifier(n_estimators=10) # n_estimators is the hyperparameter

rfe = RFE(model, n_features_to_select=15) # n_features_to_select is chosen on a trial and error basis
fit = rfe.fit(X_train, y_train)

# Get evaluation values
# print("Num Features: %s" % (fit.n_features_)) # Number of features
# print("Selected Features: %s" % (fit.support_))  # Get index .any()
# print("Feature Ranking: %s" % (fit.ranking_))
selected_features = Xi.columns[(fit.get_support())]  # Get feature column names
# print(selected_features)

 
# Apply best features to our data
Xi = Xi.drop(columns=['prectot', 't2mwet', 'ws10m_max', 'ws10m_min', 'ws50m_min', 'month'])

X_train, X_test, y_train, y_test = train_test_split(Xi, y, test_size=0.2, random_state=0)

# print("Train features shape", X_train.shape)
# print("Train target shape", y_train.shape)
# print("Test features shape", X_test.shape)
# print("Test target shape", y_test.shape)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

 
sm = SMOTE(random_state = 5)
X_train_ures_SMOTE, y_train_ures_SMOTE = sm.fit_resample(X_train, y_train)

 
# print('Before OverSampling, the shape of train_X: {}'.format(X_train.shape))
# print('Before OverSampling, the shape of train_y: {} \n'.format(y_train.shape))

# print('After OverSampling, the shape of train_X: {}'.format(X_train_ures_SMOTE.shape))
# print('After OverSampling, the shape of train_y: {} \n'.format(y_train_ures_SMOTE.shape))

# print("Counts of label '0' - Before Oversampling:{}, After OverSampling: {}".format(sum(y_train == 0),sum(y_train_ures_SMOTE == 0)))
# print("Counts of label '1' - Before Oversampling:{}, After OverSampling: {}".format(sum(y_train == 1),sum(y_train_ures_SMOTE == 1)))
# print("Counts of label '2' - Before Oversampling:{}, After OverSampling: {}".format(sum(y_train == 2),sum(y_train_ures_SMOTE == 2)))
# print("Counts of label '3' - Before Oversampling:{}, After OverSampling: {}".format(sum(y_train == 3),sum(y_train_ures_SMOTE == 3)))
# print("Counts of label '4' - Before Oversampling:{}, After OverSampling: {}".format(sum(y_train == 4),sum(y_train_ures_SMOTE == 4)))
# print("Counts of label '5' - Before Oversampling:{}, After OverSampling: {}".format(sum(y_train == 5),sum(y_train_ures_SMOTE == 5)))

 
# Create our DT Classifier with the default criterion
DT_classifier_SMOTE = tree.DecisionTreeClassifier(criterion='gini', max_depth=70)
# Fit to our SMOTE enhanced dataframe
DT_classifier_SMOTE.fit(X_train_ures_SMOTE,y_train_ures_SMOTE)
# Predict on the X_test dataset
y_pred_SMOTE = DT_classifier_SMOTE.predict(X_test)

 
# Save our DT Classifier with Pickle for further use
pickle.dump(DT_classifier_SMOTE, open('DT_classifier_SMOTE.pkl', 'wb'))

 
# Info plot gets us the model accuracy score, classification report and a visualized confusion matrix for easier observation
def info_plot(test, pred, model):
    accuracy = accuracy_score(test, pred)
   # # print("-"*55)
   # # print(f"Model Accuracy: {accuracy:.2f}")
   # # print("-"*55)
   # # print(classification_report(test, pred))
   # # print("-"*55)
    plt.figure(figsize=(8,6))
    cm = confusion_matrix(test, pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(f"Confusion Matrix")
    plt.show()


# Expects the model, train and test sets for X and y, for calculating the Cross Validation Score calculation.
# Also runs info_plot, this is why we request y_test even though we don't us it in this function

def evaluate_model(model, X_train, y_train, X_test, y_test):
    # get K-Fold CV scores
    cross_val_avg = cross_val_score(estimator=model, X=X_train, y=y_train, cv=3, scoring="accuracy")
   # # print(f"Cross-Validation Accuracy: {cross_val_avg.mean():.2f} ± {cross_val_avg.std():.2f}")

    # calculate y_pred here for infoplot
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    info_plot(y_test, y_pred, model)

# Slice our dataset into sub
X_train_ures_SMOTE_sub = X_train_ures_SMOTE[:100000]
y_train_ures_SMOTE_sub = y_train_ures_SMOTE[:100000]
X_test_sub = X_test[:100000]
y_test_sub = y_test[:100000]

evaluate_model(DT_classifier_SMOTE, X_train_ures_SMOTE_sub, y_train_ures_SMOTE_sub, X_test_sub, y_test_sub)

def smote_roc_multic():
    fpr = dict()
    tpr = dict()
    thresh = dict()

    # Define colors for the 6 classes
    colors = ['orangered', 'green', 'blue', 'yellow', 'purple', 'magenta']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i in range(6):
        fpr[i], tpr[i], thresh[i] = roc_curve(y_test, y_pred_SMOTE, pos_label=i)
        ax.plot(fpr[i], tpr[i], linestyle='--', color=colors[i], label=f'Class {i} vs Rest')

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive rate')
    ax.legend(loc='best')
    
    return fig

 
knn_classifier = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski') # default metric = "minkowski"
knn_classifier.fit(X_train, y_train)
y_pred_knn = knn_classifier.predict(X_test)

def knn_roc_multic():
    fpr = dict()
    tpr = dict()
    thresh = dict()

    # Define colors for the 6 classes
    colors = ['orangered', 'green', 'blue', 'yellow', 'purple', 'magenta']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i in range(6):
        fpr[i], tpr[i], thresh[i] = roc_curve(y_test, y_pred_knn, pos_label=i)
        ax.plot(fpr[i], tpr[i], linestyle='--', color=colors[i], label=f'Class {i} vs Rest')

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive rate')
    ax.legend(loc='best')
    
    return fig

 
k_range = list(range(1, 3)) # usually range = 1-12 but it's slow
param_grid = dict(n_neighbors=k_range)

# Apply GridSearchCV to our model to find the best parameters
grid = GridSearchCV(knn_classifier, param_grid, cv=3, scoring='accuracy', return_train_score=False,verbose=1)
grid_search=grid.fit(X_train, y_train)

 
# Get Cross Validation scores

# Create a new Dataframe and use the n_largest(5) on the test_score to analyze
score_df = pd.DataFrame(grid_search.cv_results_)
score_df.nlargest(5,"mean_test_score")

 
#Create a copy of X for ease of use
X = Xi.copy()

 
kmeans = MiniBatchKMeans(n_clusters=3, batch_size=1000) #default batch_size=100

kmeans.fit(X)
labels = kmeans.predict(X)

 
np.unique(labels, return_counts=True)

 
# Scaling our dataset without labels
X_scaled = sc.fit_transform(X)
# Slice our X in a new DataFrame X_scaled_sub
X_scaled_sub = X_scaled[:100000]

 
# To calculate the processing time, we initialize a stopwatch
start_time = time.time()

silhouettes = []
ks = list(range(2, 8))
for n_cluster in ks:
    kmeans = MiniBatchKMeans(n_clusters=n_cluster, verbose=0).fit(X_scaled_sub)
    label = kmeans.labels_
    sil_coeff = silhouette_score(X_scaled_sub, label, metric='euclidean')
   # # print("For n_clusters={}, The Silhouette Coefficient is {}".format(n_cluster, sil_coeff))
    silhouettes.append(sil_coeff)

plt.figure(figsize=(12, 8))
plt.subplot(211)
plt.scatter(ks, silhouettes, marker='x', c='r')
plt.plot(ks, silhouettes)
plt.xlabel('k')
plt.ylabel('Silhouette score')

# How long did it take to run? Here's the time in seconds
# print("Time taken: {:.2f} seconds".format(time.time() - start_time))


