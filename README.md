# NeckPain

### Abstract 
Objective: Neck pain conditions (NP) are among the most prevalent diseases globally, and a leading cause of disability, healthcare visits, and lost productivity. Their heterogeneity must be better understood to improve clinical management. Materials and Methods: EHR data (N=227,130) were obtained from adult patients (18-110 years old) in the Yale New Haven Health System, with an incident NP related encounter between 10/1/2016 to 09/30/2023. We conducted clustering analysis on NP populations to characterize heterogeneous NP patterns and identify novel subphenotypes and evaluated how these patterns were associated with subsequent prescription opioid use and healthcare utilization. Results: A four-cluster solution allowed for clinical interpretability with meaningful age groupings separated by comorbidity burden including mental health diagnoses, physical functioning, and sociodemographic variables. CT scanning/imaging, prescription opioid use outside of surgery, inpatient hospitalization, and injection/surgery were significantly higher in the cluster of oldest mean age (Cluster 3) with the highest rates of life-limiting, depression and anxiety comorbidities. Significantly higher ER use was observed in younger persons with low comorbidity rates (Cluster 4), with lower rates of interventions. Discussion: The association of subphenotypes and outcomes suggests that comorbidity patterns and sociodemographic factors, in addition to age, should be considered in treatment decisions to reduce negative events including opioid prescription. Conclusions: Neck pain subphenotypes are important for both clinical care and research. The observed low rates of injections suggest more emphasis be placed on these treatment options in education and guidelines. The findings also suggest the need to better connect ER users with neck pain to follow-up services to potentially reduce transition to chronic pain. 

 ### Keywords
 Neck Pain; Machine Learning; Patient Outcome Assessment; Clinical Subphenotypes; Cohort Study 

## Analysis pipeline

The scripts in [`code/`](code/) reproduce the analyses reported in the paper. They are numbered in the order they are run; each step reads the outputs of the previous one.

| Script | Purpose |
|---|---|
| `1_data_input.py` | Assemble the cohort, recode demographics, group ICD-based comorbidity flags, merge the Charlson (CCI) and Functional (FCI) comorbidity indices, and write the patient-level feature table. |
| `2_kmeans_cluster.py` | K-means clustering (one-hot + standardized features), K selection (elbow + silhouette), and stability assessment (random-seed ARI, bootstrap ARI, cluster-wise Jaccard). |
| `3_diff_test.py` | Compare baseline characteristics across clusters (Kruskal–Wallis for continuous, chi-square / Fisher for categorical/binary) with Benjamini–Hochberg FDR correction. |
| `4_data_outcome.py` | Summarize outcomes by cluster: healthcare visit utilization (emergency / inpatient / outpatient) and prescription opioid use. |
| `5_procedures_analysis.py` | Procedure utilization by cluster (imaging and intervention categories): per-cluster rates, cross-cluster chi-square with BH-FDR, and summary figures. |
| `6_opioid_surgery_sensitivity.py` | Sensitivity analysis for opioid use: opioid × surgery cross-tabulation per cluster, re-analysis excluding surgical patients, and a surgery-adjusted logistic model. |

## Data availability

The analyses use patient-level electronic health record data from the Yale New Haven Health System under an approved IRB protocol. These data contain protected health information and **cannot be shared publicly**. Input data paths in the scripts are placeholders (e.g., `path/to/...`); replace them with the location of your own OMOP-formatted source files to run the pipeline.

## Requirements

Python 3.10+. Install dependencies with:

```
pip install -r requirements.txt
```
