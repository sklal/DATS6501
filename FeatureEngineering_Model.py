############################################Libraries########################################################
from Prescriber import *
from Payment import *
from LEIE import *
print(Final.head())
Final.fillna(0, inplace=True)

print(Final.isnull().sum())
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import brier_score_loss, precision_score, recall_score,f1_score, roc_auc_score, accuracy_score
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
########################################## Scaling the features############################################################

print(Final['total_day_supply_sum'].dtype)
# Scaling the features
Final['total_drug_cost_sum'] = Final['total_drug_cost_sum'].map(lambda x: np.log10(x + 1.0))
Final['total_claim_count_sum'] = Final['total_claim_count_sum'].map(lambda x: np.log10(x + 1.0))
Final['total_day_supply_sum'] = Final['total_day_supply_sum'].map(lambda x: np.log10(x + 1.0))
Final['Total_Payment_sum'] = Final['Total_Payment_sum'].map(lambda x: np.log10(x + 1.0))

Final['total_drug_cost_mean'] = Final['total_drug_cost_mean'].map(lambda x: np.log10(x + 1.0))
Final['total_claim_count_mean'] = Final['total_claim_count_mean'].map(lambda x: np.log10(x + 1.0))
Final['total_day_supply_mean'] = Final['total_day_supply_mean'].map(lambda x: np.log10(x + 1.0))

Final['total_drug_cost_max'] = Final['total_drug_cost_max'].map(lambda x: np.log10(x + 1.0))
Final['total_claim_count_max'] = Final['total_claim_count_max'].map(lambda x: np.log10(x + 1.0))
Final['total_day_supply_max'] = Final['total_day_supply_max'].map(lambda x: np.log10(x + 1.0))


Final['claim_max-mean'] = Final['total_claim_count_max'] - Final['total_claim_count_mean']

Final['supply_max-mean'] = Final['total_day_supply_max'] - Final['total_day_supply_mean']

Final['drug_max-mean'] = Final['total_drug_cost_max'] - Final['total_drug_cost_mean']


Final['npi'] = Final.npi.astype(object)

categorical_features = ['npi','last_name', 'Specialty','first_name','city', 'state']

numerical_features = ['total_drug_cost_sum', 'total_drug_cost_mean','Total_Payment_sum',
       'total_drug_cost_max', 'total_claim_count_sum',
       'total_claim_count_mean', 'total_claim_count_max',
       'total_day_supply_sum', 'total_day_supply_mean', 'total_day_supply_max',
    'claim_max-mean','supply_max-mean', 'drug_max-mean']

target = ['is_fraud']
#Combining all the variables
allvars = categorical_features + numerical_features + target

y = Final["is_fraud"].values
X = Final[allvars].drop('is_fraud',axis=1)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=42)
print(X_train.shape)
print(X_valid.shape)

X_train[numerical_features] = X_train.loc[:,numerical_features].fillna(0)
X_valid[numerical_features] = X_valid.loc[:,numerical_features].fillna(0)
X_train[categorical_features] = X_train.loc[:,categorical_features].fillna('NA')
X_valid[categorical_features] = X_valid.loc[:,categorical_features].fillna('NA')


scaler= StandardScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features].values)
X_valid[numerical_features] = scaler.transform(X_valid[numerical_features].values)


########################################Train-Test Split############################################
df_train, df_test = train_test_split(Final, test_size=0.3)

partD_drug_train = pd.merge(partD_drugs,df_train[['npi','is_fraud']], how='inner', on=['npi'])
partD_drug_All = pd.merge(partD_drugs,Final[['npi','is_fraud']], how='inner', on=['npi'])


#Incorporating drugs to our feature
drugs = set([ drugx for drugx in partD_drug_train['drug_name'].values if isinstance(drugx, str)])
print(len(drugs))

print("Total records in train set : ")
print(len(partD_drug_train))
print("Total Fraud in train set : ")
print(len(partD_drug_train[partD_drug_train['is_fraud']==1]))
partD_drug_train.head()

cols = ['total_drug_cost','total_claim_count','total_day_supply']

partD_drug_train_Group = partD_drug_train.groupby(['drug_name', 'is_fraud'])
partD_drug_All_Group = partD_drug_All.groupby(['drug_name', 'is_fraud'])

drug_keys = partD_drug_train_Group.groups.keys()
print(len(drug_keys))

drug_with_isfraud = [drugx for drugx in drugs if ((drugx,0.0) in drug_keys ) & ( (drugx,1.0) in drug_keys)]


# T-Test on drug group
from scipy.stats import ttest_ind
re_drug_tt = dict()
for drugx in drug_with_isfraud:
    for colx in cols:
        fraud_0 = partD_drug_train_Group.get_group((drugx,0.0))[colx].values
        fraud_1 = partD_drug_train_Group.get_group((drugx,1.0))[colx].values
        # print len(fraud_0), len(fraud_1)
        if (len(fraud_0)>2) & (len(fraud_1)>2) :
            tt = ttest_ind(fraud_0, fraud_1)
            re_drug_tt[(drugx, colx)] = tt



#Setting Probabilities
Prob_005 = [(key, p) for (key, (t, p)) in re_drug_tt.items() if p <=0.05]
print(len(Prob_005))

inx=100
drug_name = Prob_005[inx][0][0]
print(drug_name)
df_bar = pd.concat([partD_drug_All_Group.get_group((Prob_005[inx][0][0],0.0)), partD_drug_All_Group.get_group((Prob_005[inx][0][0],1.0))])
df_bar.head()

Feature_DrugWeighted = []
new_col_all = []
for i, p005x in enumerate(Prob_005):
    # if i>4:
    #   break
    drug_name = p005x[0][0]
    cat_name = p005x[0][1]

    new_col = drug_name + '_' + cat_name
    new_col_all.append(new_col)

    drug_0 = partD_drug_All_Group.get_group((drug_name, 0.0))[['npi', cat_name]]
    drug_1 = partD_drug_All_Group.get_group((drug_name, 1.0))[['npi', cat_name]]

    drug_01 = pd.concat([drug_0, drug_1])
    drug_01.rename(columns={cat_name: new_col}, inplace=True)
    Feature_DrugWeighted.append(drug_01)

npi_col = Final[['npi']]

w_npi = []

for n, nx in enumerate(Feature_DrugWeighted):
    nggx = pd.merge(npi_col, nx.drop_duplicates(['npi']), on='npi', how='left')

    w_npi.append(nggx)

Final1 = Final

for wx in w_npi:
    col_n = wx.columns[1]
    Final1[col_n] = wx[col_n].values

wx = w_npi[0]
wx.columns[1]
col_n = wx.columns[1]

len(wx[col_n].values)
Final1.fillna(0)

new_col_all

print(Final1[new_col_all].describe())

Final1['drug_mean'] = Final1[new_col_all].mean(axis=1)


Final['drug_mean'] = Final['drug_mean'].map(lambda x: np.log10(x + 1.0))

Final1['drug_sum'] = Final1[new_col_all].sum(axis=1)
Final['drug_sum'] = Final['drug_sum'].map(lambda x: np.log10(x + 1.0))

Final1['drug_variance'] = Final1[new_col_all].var(axis=1)

df_train, df_valid = train_test_split(Final1, test_size=0.3)

df_train.fillna(0)
df_valid.fillna(0)

#Create the Specialty Weight
spec_dict =[]
spec_fraud_1 = df_train[df_train['is_fraud']==1]['Specialty']

from collections import Counter
counts = Counter(spec_fraud_1)
spec_dict =  dict(counts)

Final1['Spec_Weight'] = Final1['Specialty'].map(lambda x: spec_dict.get(x, 0))

df_train, df_valid = train_test_split(Final1, test_size=0.3)

numerical_features1 = numerical_features + ['drug_sum','Spec_Weight']

import seaborn as sns

# Default heatmap
# Calculate correlation between each pair of variable
corr_matrix = df_train.corr()

# Draw the heatmap with the mask
#sns.heatmap(corr_matrix)

X= df_train[numerical_features1].values
Y = df_train['is_fraud'].values

params_0 = {'n_estimators': 300, 'max_depth': 6, 'min_samples_split': 3, 'learning_rate': 0.01}
params_1 = {'n_estimators': 500, 'max_depth': 10, 'min_samples_split': 5, 'class_weight': {0: 1, 1: 2000}, 'n_jobs': 5}

scaler = StandardScaler()

clfs = [
    LogisticRegression(C=1e5, class_weight={0: 1, 1: 2000}, n_jobs=5),

    GaussianNB(),

    RandomForestClassifier(**params_1),

    GradientBoostingClassifier(**params_0)

]

X_train = df_train[numerical_features1].values

y_train = df_train['is_fraud'].values

X_train = scaler.fit_transform(X_train)

X_valid = df_valid[numerical_features1].values
y_valid = df_valid['is_fraud'].values
X_valid_x = scaler.transform(X_valid)

prob_result = []
df_m = []
clfs_fited = []
for clf in clfs:
    print("%s:" %  clf.__class__.__name__)
    clf.fit(X_train,y_train)
    clfs_fited.append(clf)
    y_pred = clf.predict(X_valid_x)
    prob_pos  = clf.predict_proba(X_valid_x)[:, 1]
    prob_result.append(prob_pos)
    m = confusion_matrix(y_valid, y_pred)
    clf_score = brier_score_loss(y_valid, prob_pos, pos_label=y_valid.max())
    print("\tBrier: %1.5f" % (clf_score))
    print("\tPrecision: %1.5f" % precision_score(y_valid, y_pred))
    print("\tRecall: %1.5f" % recall_score(y_valid, y_pred))
    print("\tF1: %1.5f" % f1_score(y_valid, y_pred))
    print("\tauc: %1.5f" % roc_auc_score(y_valid, prob_pos))
    print("\tAccuracy: %1.5f\n" % accuracy_score(y_valid, y_pred))
    df_m.append(
        pd.DataFrame(m, index=['True Negative', 'True Positive'], columns=['Pred. Negative', 'Pred. Positive'])
        )


fpr, tpr, thresholds = roc_curve(y_valid, prob_result[2])

fpr, tpr, thresholds = roc_curve(y_valid, prob_result[2])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.2f)' % roc_auc)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

print(m)

y_pred = clf.predict(X_valid_x)

feature_importance = clfs_fited[2].feature_importances_
# make importance relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)

features = [numerical_features1[ix] for ix in sorted_idx]
bardata = {"name":features[::-1], "importance percent":feature_importance[sorted_idx][::-1]}

plt.figure()

# Create plot title
plt.title("Feature Importance")

# Add bars
plt.bar(range(X.shape[1]), feature_importance[sorted_idx])

# Add feature names as x-axis labels
plt.xticks(range(X.shape[1]), features, rotation=90)

# Show plot
plt.show()