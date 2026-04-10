# Somatic-mutations-and-breast-cancer-prognosis
ML pipeline to predict 5-year breast cancer survival from somatic mutations in TCGA-BRCA patients. 

*Context*

Breast cancer accounts for approximately 30% of all female cancer diagnoses, with an estimated 2.3 million new cases each year (Sung et al., 2021).

TCGA-BRCA is a public dataset that collects tumor mutation data, gene activity levels, methylation and clinical records for over 1000 breast cancer patients. Several research groups have used this resource to build prognostic models. However, a consistent finding is that mutation data alone tends to be one of the weakest predictors of survival. 

Despite this, looking at mutations is still worthwhile as certain genes like TP53, PIK3CA, CDH1 GATA3 and BRCA1/2 have well known links to patient outcomes (Pang et al., 2014; Meric-Bernstan et al., 2018).

*Stack*
- Language: Python
- Data: Somatic mutation files and clinical records from the TCGA-BRCA
- Libraries: pandas, numpy, scikit-learn, xgboost, shap, matplotlib, seaborn, requests

*Methods*
1. Data collection: Somatic mutation files and clinical records for TCGA-BRCA patients were downloaded using the NCI Genomic Data Commons public API. The clinical data included age at diagnosis, tumor stage and survival information. A patient was considered a 5-year survivor if their follow-up or time of death was 1825 days or more.
2. Annotation: Each mutation was labeled with biological information, such as how damaging it is, what type of variant it is and whether it affects a known cancer gene or biological process like DNA repair or cell division. 
3. Model: Three classifiers were tested, Logistic Regression, Random Forest and XGBoost. SHAP values were then used to understand which features had the most influence on the predictions.

*Results*

Logistic Regression performed best with a test AUC of 0.55, followed by Random Forest (0.51) and XGBoost (0.50). All three models performed fairly low, which is expected when working with mutation data alone.

SHAP analysis showed that maximum variant allele frequency, age at diagnosis and mean mutation impact score were the most influential features. High impact mutations such as truncating and nonsense variants also ranked highly.

TP53 mutations were slightly more frequent in non-survivors, consistent with its established role as a negative prognostic marker in breast cancer. PIK3CA mutations were slightly more common in survivors, in line with previous findings linking PIK3CA to better outcomes in certain patient groups.

The main limitation is that the model only used mutation data. Studies that incorporate gene activity levels and clinical variables such as tumor size and lymph node involvement consistently achieve higher prediction accuracy.

*References*

Meric-Bernstam, F., Zheng, X., Shariati, M., Damodaran, S., Wathoo, C., Brusco, L., Demirhan, M. E., Tapia, C., Eterovic, A. K., Basho, R. K., Ueno, N. T., Janku, F., Sahin, A., Rodon, J., Broaddus, R., Kim, T.-B., Mendelsohn, J., Mills Shaw, K. R., Tripathy, D., … Chen, K. (2018). Survival outcomes by TP53 mutation status in metastatic breast cancer. JCO Precision Oncology, 2018(2), 1–15. https://doi.org/10.1200/PO.17.00245

Pang, B., Cheng, S., Sun, S.-P., An, C., Liu, Z.-Y., Feng, X., & Liu, G.-J. (2014). Prognostic role of PIK3CA mutations and their association with hormone receptor expression in breast cancer: a meta-analysis. Scientific Reports, 4(1), 6255. https://doi.org/10.1038/srep06255

Sung, H., Ferlay, J., Siegel, R. L., Laversanne, M., Soerjomataram, I., Jemal, A., & Bray, F. (2021). Global cancer statistics 2020: GLOBOCAN estimates of incidence and mortality worldwide for 36 cancers in 185 countries. CA: A Cancer Journal for Clinicians, 71(3), 209–249. https://doi.org/10.3322/caac.21660
