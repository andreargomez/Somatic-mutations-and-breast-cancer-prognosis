import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, classification_report,
    confusion_matrix, RocCurveDisplay
)
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import shap
import warnings
import os

os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("data/brcaa.csv")
df = df[df['survival_5yr'].notna()].copy()

df['survival_5yr'] = df['survival_5yr'].astype(int)

featcols = [
    'total_mutations', 'mean_impact_score', 'max_impact_score',
    'n_high_impact', 'n_cancer_gene_muts', 'n_oncogene_muts',
    'n_tsg_muts', 'n_missense', 'n_truncating', 'n_snp', 'n_indel',
    'mean_vaf', 'max_vaf', 'n_pi3k_muts', 'n_dna_repair_muts',
    'n_cell_cycle_muts', 'n_rtk_ras_muts', 'n_hormone_muts',
    'has_TP53', 'has_PIK3CA', 'has_CDH1', 'has_BRCA1', 'has_BRCA2',
    'has_GATA3', 'has_PTEN', 'has_MAP3K1',
]

df['tumor_stage_num'] = pd.to_numeric(df['tumor_stage'], errors='coerce')
featcols.append('tumor_stage_num')
featcols.append('age_at_diagnosis')

X = df[featcols].copy()
y = df['survival_5yr']

X = X.fillna(X.median())
X = X.fillna(0)

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

ytrain = ytrain.astype(int)
ytest = ytest.astype(int)

scaler = StandardScaler()
Xtrains = scaler.fit_transform(Xtrain)
Xtests = scaler.transform(Xtest)

cls = np.array(sorted(ytrain.unique()))
wts = compute_class_weight('balanced', classes=cls, y=ytrain)
cwdict = dict(zip(cls, wts))
print(f"Class weights: {cwdict}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# logistic regression
lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lrcv = cross_val_score(lr, Xtrains, ytrain, cv=cv, scoring='roc_auc')
lr.fit(Xtrains, ytrain)
lrauc = roc_auc_score(ytest, lr.predict_proba(Xtests)[:, 1])
print(f"  CV AUC: {lrcv.mean():.3f} ± {lrcv.std():.3f}")
print(f"  Test AUC: {lrauc:.3f}")

# random forest
rf = RandomForestClassifier(
    n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1
)
rfcv = cross_val_score(rf, Xtrain, ytrain, cv=cv, scoring='roc_auc')
rf.fit(Xtrain, ytrain)
rfauc = roc_auc_score(ytest, rf.predict_proba(Xtest)[:, 1])
print(f"  CV AUC: {rfcv.mean():.3f} ± {rfcv.std():.3f}")
print(f"  Test AUC: {rfauc:.3f}")

# xgboost
spw = (ytrain == 0).sum() / (ytrain == 1).sum()
xgbm = xgb.XGBClassifier(
    n_estimators=200, scale_pos_weight=spw,
    random_state=42, eval_metric='auc', verbosity=0
)
xgbcv = cross_val_score(xgbm, Xtrain, ytrain, cv=cv, scoring='roc_auc')
xgbm.fit(Xtrain, ytrain)
xgbauc = roc_auc_score(ytest, xgbm.predict_proba(Xtest)[:, 1])
print(f"  CV AUC: {xgbcv.mean():.3f} ± {xgbcv.std():.3f}")
print(f"  Test AUC: {xgbauc:.3f}")

bname, bmodel, bauc, bXtest = max(
    [
        ('Logistic Regression', lr, lrauc, Xtests),
        ('Random Forest', rf, rfauc, Xtest),
        ('XGBoost', xgbm, xgbauc, Xtest)
    ],
    key=lambda x: x[2]
)
print(f"\nBest model: {bname} (AUC = {bauc:.3f})")

ypred = bmodel.predict(bXtest)
print(classification_report(ytest, ypred, target_names=['Did not survive 5yr', 'Survived 5yr']))

# ROC curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
for name, model, Xt in [
    ('Logistic Regression', lr, Xtests),
    ('Random Forest', rf, Xtest),
    ('XGBoost', xgbm, Xtest)
]:
    RocCurveDisplay.from_estimator(model, Xt, ytest, ax=ax, name=name)

ax.plot([0, 1], [0, 1], 'k--', label='Random classifier')
ax.set_title('ROC Curves — All Models', fontsize=13)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(loc='lower right')

ax2 = axes[1]
if bname in ['Random Forest', 'XGBoost']:
    imps = pd.Series(
        bmodel.feature_importances_,
        index=featcols
    ).sort_values(ascending=True).tail(15)
    imps.plot(kind='barh', ax=ax2, color='steelblue')
    ax2.set_title(f'Top 15 Feature Importances\n({bname})', fontsize=13)
    ax2.set_xlabel('Importance Score')
else:
    coefs = pd.Series(
        np.abs(lr.coef_[0]),
        index=featcols
    ).sort_values(ascending=True).tail(15)
    coefs.plot(kind='barh', ax=ax2, color='steelblue')
    ax2.set_title('Top 15 Feature Coefficients\n(Logistic Regression)', fontsize=13)

plt.tight_layout()
plt.savefig("outputs/model_results.png", dpi=150, bbox_inches='tight')
plt.show()

# shap
exp = shap.TreeExplainer(xgbm)
shapvals = exp.shap_values(Xtest)

plt.figure(figsize=(10, 8))
shap.summary_plot(shapvals, Xtest, feature_names=featcols, show=False, plot_size=None)
plt.title("SHAP Feature Impact on 5-Year Survival Prediction\n(TCGA-BRCA, n=917)",
          fontsize=12, pad=15)
plt.tight_layout()
plt.savefig("outputs/shap_beeswarm.png", dpi=150, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 7))
shap.summary_plot(shapvals, Xtest, feature_names=featcols,
                  plot_type="bar", show=False, plot_size=None)
plt.title("Mean Absolute SHAP Values — Feature Importance\n(TCGA-BRCA XGBoost)",
          fontsize=12, pad=15)
plt.tight_layout()
plt.savefig("outputs/shap_bar.png", dpi=150, bbox_inches='tight')
plt.show()

# mutation burden plots
plotdf = df[['survival_5yr', 'total_mutations', 'n_cancer_gene_muts',
             'has_TP53', 'has_PIK3CA', 'has_BRCA1', 'has_BRCA2']].copy()
plotdf['survival_5yr'] = plotdf['survival_5yr'].map(
    {0: 'Did not survive 5yr', 1: 'Survived 5yr'}
)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Mutation Patterns: Survivors vs Non-Survivors\n(TCGA-BRCA)',
             fontsize=13, y=1.02)

ax1 = axes[0]
surv = plotdf[plotdf['survival_5yr'] == 'Survived 5yr']['total_mutations']
nsurv = plotdf[plotdf['survival_5yr'] == 'Did not survive 5yr']['total_mutations']
ax1.hist(nsurv.clip(upper=200), bins=30, alpha=0.6,
         color='salmon', label='Did not survive 5yr', density=True)
ax1.hist(surv.clip(upper=200), bins=30, alpha=0.6,
         color='steelblue', label='Survived 5yr', density=True)
ax1.set_xlabel('Total Mutations (capped at 200)')
ax1.set_ylabel('Density')
ax1.set_title('Mutation Burden Distribution')
ax1.legend(fontsize=8)

ax2 = axes[1]
sns.boxplot(
    data=plotdf,
    x='survival_5yr',
    y='n_cancer_gene_muts',
    hue='survival_5yr',
    palette={'Did not survive 5yr': 'salmon', 'Survived 5yr': 'steelblue'},
    legend=False,
    ax=ax2
)
ax2.set_title('Cancer Gene Mutation Count')
ax2.set_xlabel('')
ax2.set_ylabel('# Mutations in Cancer Genes')
ax2.tick_params(axis='x', labelsize=8)

ax3 = axes[2]
genes = ['has_TP53', 'has_PIK3CA', 'has_BRCA1', 'has_BRCA2']
glabels = ['TP53', 'PIK3CA', 'BRCA1', 'BRCA2']
srates = [plotdf[plotdf['survival_5yr'] == 'Survived 5yr'][g].mean() * 100 for g in genes]
nsrates = [plotdf[plotdf['survival_5yr'] == 'Did not survive 5yr'][g].mean() * 100 for g in genes]
x = np.arange(len(genes))
width = 0.35
ax3.bar(x - width/2, nsrates, width, label='Did not survive 5yr',
        color='salmon', alpha=0.8)
ax3.bar(x + width/2, srates, width, label='Survived 5yr',
        color='steelblue', alpha=0.8)
ax3.set_xticks(x)
ax3.set_xticklabels(glabels)
ax3.set_ylabel('% of Patients with Mutation')
ax3.set_title('Key Gene Mutation Rates\nby Survival Group')
ax3.legend(fontsize=8)

plt.tight_layout()
plt.savefig("outputs/biological_validation.png", dpi=150, bbox_inches='tight')
plt.show()

# save predictions
resdf = Xtest.copy()
resdf['true_label'] = ytest.values
resdf['predicted_survival_prob'] = xgbm.predict_proba(Xtest)[:, 1]
resdf['predicted_label'] = xgbm.predict(Xtest)
resdf.to_csv("outputs/predictions.csv", index=False)
print("done!!!")
