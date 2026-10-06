import duckdb, os
import pandas as pd
import numpy as np

path = 'path/to/patient_cohort/'
patient_index_file = 'patient_index_demo.parquet'
comorbidities_file = 'comorbidities.csv'
charlson_index_file = 'Charlson_result.csv'
comorbidity_index_file = 'ComorbidityIndex_result.csv'

df_patient_index = duckdb.sql(f"""
SELECT person_id, age, ethnicity_source_value, race_source_value, gender_source_value, TreatableNP, RadicularNP, GenericNP, SeriousNP
FROM "{path}/{patient_index_file}"
-- LIMIT 10
""").df()

df_patient_index = df_patient_index.replace([
    None,
    "",
    "I Do Not Know",
    "I Prefer Not To Share",
    "I Do Not See My Race Listed Here",
    "Other",
    "Unknown",
    "X",
    "Unable to Assess"
], "unknown/private/missing")

df_patient_index = df_patient_index.replace([
    "Puerto Rican",
    "Mexican, Mexican American, Chicano/a",
    "Cuban"
], "Hispanic or Latino/a/e")
df_patient_index = df_patient_index.replace([
    "American Indian or Native American",
    "Pacific Islander",
    "Native Hawaiian or Other Pacific Islander"
], "American Indian or Native American or Pacific Islander or Native Hawaiian or Other Pacific Islander")
df_patient_index = df_patient_index.replace([
    "Filipino",
    "Chinese",
    "Korean",
    "Japanese",
    "Vietnamese"
], "Asian")
df_patient_index = df_patient_index.replace([
    "Asian",
    "American Indian or Native American or Pacific Islander or Native Hawaiian or Other Pacific Islander",
    "Middle Eastern or Northern African",
    "Native Hawaiian",
], "Other")

df_patient_index = df_patient_index.replace([
    "Single Adult",
    "Single Minor",
    "Never married"
], "Single")
df_patient_index = df_patient_index.replace([
    "divorced",
    "seperated"
], "Divorced")
df_patient_index = df_patient_index.replace([
    "married",
    "Committed Relationship",
    "Long-term committed relationship",
    "Long-term commiited relationship",
    "Civil Union",
    "Long-term committed relationship",
    "Partnered"
], "Married")
df_patient_index = df_patient_index.replace([
    "widowed",
], "Widowed")

df_patient_index

fp = os.path.join(path, comorbidities_file)
df = pd.read_csv(fp, sep=",")

prevalence_rate = df.iloc[:,1:].mean() * 100
summary_df = pd.DataFrame(prevalence_rate, columns=['Prevalence Rate (%)'])
sorted_summary_df = summary_df.sort_values(by=summary_df.columns[0], ascending=False)
print(sorted_summary_df)

chronic_pain_cols = [
    "KAISER_AbdominalPain",
    "KAISER_BoneInfection",
    "KAISER_fibromyalgia",
    "KAISER_InfArthriticDis",
    "KAISER_LimbPain",
    "KAISER_MusChestPain",
    "KAISER_Neuropathy",
    "KAISER_OtherPain",
    "KAISER_PelvicPain",
    "KAISER_SystemicPain",
    "MSD_JointDisorders",
    "MSD_NonTraumatic_jd",
    "MSD_Osteomyl",
    "MSD_RA",
    "MSD_SpinalCord_inj",
    "KAISER_TMD"
]
df_comorbidities = df[["person_id"] + chronic_pain_cols].copy()
df_comorbidities["chronic_pain_conditions"] = df[chronic_pain_cols].sum(axis=1)

HCOE_AnyHA_cols = [
    "HCOE_Migraine",
    "HCOE_TensionHA",
    "HCOE_PostTraumaHA",
    "HCOE_PostWhiplashHA",
    "HCOE_OtherPrimaryHA",
    "HCOE_TAC",
    "HCOE_SecondaryHA",
    "HCOE_HeadacheNOS"
]
df_comorbidities["HCOE_AnyHA"] = df[HCOE_AnyHA_cols].max(axis=1)

Depressive_cols = [
    "NEPEC_MDD",
    "NEPEC_DXODep",
    "NEPEC_AFBPDx"
]
df_comorbidities["Depressive"] = df[Depressive_cols].max(axis=1)

Anxiety_cols = [
    "NEPEC_ANXunsp",
    "NEPEC_PTSD",
    "NEPEC_ANXgen"
]
df_comorbidities["Anxiety"] = df[Anxiety_cols].max(axis=1)

SleepDxs_cols = [
    "NEPEC_INSOMNIA",
    "NEPEC_BREATHINGDO",
    "HCOE_SleepApnea",
    "NEPEC_NIGHTMARES"
]
df_comorbidities["SleepDxs"] = df[SleepDxs_cols].max(axis=1)

Neurological_cols = [
    "HCOE_TBI",
    "HCOE_ALS",
    "HCOE_Epilepsy",
    "HCOE_PARKMOVE",
    "HCOE_PARKTREM",
    "HCOE_MS",
    "HCOE_TIA",
    "HCOE_IschemicStroke"
]
df_comorbidities["Neurological"] = df[Neurological_cols].max(axis=1)

Substances_cols = [
    "NEPEC_DXALC",
    "NEPEC_DXDRG"
]
df_comorbidities["Substances"] = df[Substances_cols].max(axis=1)

drop_cols = chronic_pain_cols + [
    "KAISER_HEADACHE",
    "KAISER_NeckPain",
    "MSD_CongenitalAnom",
    "MSD_Deformities",
    "MSD_Fracture",
    "MSD_Gout",
    "MSD_Lupus",
    "MSD_NervousSystem",
    "MSD_Osteoporosis",
    "MSD_OthBoneDisease",
    "MSD_Sprain_Strain",
    "MSD_TissueDisease",
    "NEPEC_SCHZ",
    "OpioidOverdose",
    "UW_ARDS",
    "UW_Asthma",
    "UW_Bronch",
    "UW_COPD",
    "UW_ILD",
    "UW_PneuBact",
    "UW_PneuUnsp",
    "UW_PneuVent",
    "UW_PneuViral",
    "UW_PulmHTN",
    "UW_RespFail",
    "HCOE_Headache",
    "KAISER_Headache",
    "KAISER_BackPain",
    "MSD_TMD",
]
df_comorbidities = df_comorbidities.drop(columns=drop_cols, errors="ignore")

processed_cols = (
    ["person_id"]
    + chronic_pain_cols
    + HCOE_AnyHA_cols
    + Depressive_cols
    + Anxiety_cols
    + SleepDxs_cols
    + Neurological_cols
    + Substances_cols
    + drop_cols
)
remaining_cols = df.columns.drop(processed_cols, errors="ignore")

df_comorbidities = pd.concat(
    [df_comorbidities, df[remaining_cols]],
    axis=1
)

df_comorbidities['kaiser_Fractures'] = (
    (df_comorbidities['KAISER_Fractures'] == 1) |
    (df_comorbidities['Kaiser_Fractures'] == 1)
).astype(int)

df_comorbidities.drop(columns=['KAISER_Fractures', 'Kaiser_Fractures'], inplace=True)

prevalence_rate = df_comorbidities.iloc[:,1:].mean() * 100
summary_df = pd.DataFrame(prevalence_rate, columns=['Prevalence Rate (%)'])

print(summary_df)

df_comorbidities

fp = os.path.join(path, charlson_index_file)
df = pd.read_csv(fp, sep=",")
df_charlson_index = df[["person_id", "CCI_Score"]]
df_charlson_index = df_charlson_index.fillna(0)

df_charlson_index["CCI_Score"] = np.where(
    df_charlson_index["CCI_Score"] >= 4,
    "4+",
    df_charlson_index["CCI_Score"]
)

df_charlson_index

fp = os.path.join(path, comorbidity_index_file)
df = pd.read_csv(fp, sep=",")
df_FCI_Score = df[["person_id", "FCI_Score"]]
df_FCI_Score

df_input = df_patient_index.merge(df_comorbidities, on="person_id", how="outer").merge(df_charlson_index, on="person_id", how="outer").merge(df_FCI_Score, on="person_id", how="outer")
print(f"* found {len(df_input)} patients!")

print(f"- N(patients) < 18: {sum(df_input.age < 18)}")
print(f"- N(patients) > 110: {sum(df_input.age > 110)}")

df_input = df_input[(df_input.age >= 18) & (df_input.age <= 110)]
print(f"* finalized {len(df_input)} patients!")

df_input = df_input.fillna(0)
print(df_input.isna().sum().sum())

fp = "data/data_input.tsv"
df_input.to_csv(fp, sep="\t", index=False)
print(f"* saved {len(df_input)} records to {fp}!")

numerical_columns = [
    "age",
]
categorical_columns = [
    "ethnicity_source_value",
    "race_source_value",
    "gender_source_value",

    "TreatableNP",
    "RadicularNP",
    "GenericNP",
    "SeriousNP"
]

total_rows = len(df_input)

markdown_content = "# Data Summary Report\n\n"

markdown_content += "## Numerical Columns Summary\n\n"
markdown_content += "| Column Name | Non-NA Count | Non-NA Percentage |\n"
markdown_content += "|-------------|--------------|-------------------|\n"

for col in numerical_columns:
    non_na_count = df_input[col].notna().sum()
    non_na_percentage = (non_na_count / total_rows) * 100
    markdown_content += f"| {col} | {non_na_count} | {non_na_percentage:.2f}% |\n"

markdown_content += "\n"

markdown_content += "| Column Name | Min | Mean | Median | Max |\n"
markdown_content += "|-------------|-------------|-------------|--------------|-------------------|\n"

for col in numerical_columns:
    minimum = df_input[col].min()
    mean = df_input[col].mean()
    median = df_input[col].median()
    maximum = df_input[col].max()
    markdown_content += f"| {col} | {minimum} | {mean} | {median} | {maximum} |\n"

markdown_content += "\n"

markdown_content += "## Categorical Columns Summary\n\n"

for col in categorical_columns:
    non_na_count = df_input[col].notna().sum()
    non_na_percentage = (non_na_count / total_rows) * 100

    markdown_content += f"### {col}:\n"
    markdown_content += f"Non-NA count: {non_na_count}\n\n"
    markdown_content += f"Non-NA percentage: {non_na_percentage:.2f}%\n\n"
    markdown_content += "Value Counts:\n"

    value_counts = df_input[col].value_counts(dropna=False)
    for value, count in value_counts.items():
        markdown_content += f"- {value}: {count} ({count/total_rows *100:.2f}%)\n"

    markdown_content += "\n"

output_file = "./outputs/data_summary/data_summary_report_1.md"
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, "w") as file:
    file.write(markdown_content)

print(f"Markdown summary report saved as '{output_file}'")

def summarize_integer_column(df, column_name):
    summary = {}
    summary['Count'] = df[column_name].count()
    summary['NA Count'] = df[column_name].isna().sum()
    summary['Mean'] = df[column_name].mean()
    summary['Median'] = df[column_name].median()
    summary['Standard Deviation'] = df[column_name].std()
    summary['Variance'] = df[column_name].var()
    summary['Minimum'] = df[column_name].min()
    summary['Maximum'] = df[column_name].max()

    summary['Distribution'] = df[column_name].value_counts(dropna=False).to_dict()
    return summary

def summarize_category_column(df, column_name):
    summary = {}
    summary['Count'] = df[column_name].count()
    summary['NA Count'] = df[column_name].isna().sum()

    summary['Distribution'] = df[column_name].value_counts(dropna=False).to_dict()
    return summary

def generate_markdown_report(df, output_file):
    comorb_total_na = df.iloc[:,10:-2].isna().sum().sum()
    prevalence_rate = df.iloc[:,10:-2].mean() * 100
    prevalence_df = pd.DataFrame(prevalence_rate, columns=['Prevalence Rate (%)'])

    chronic_pain_conditions_total_na = df["chronic_pain_conditions"].isna().sum().sum()
    chronic_pain_conditions_summary = summarize_integer_column(df, "chronic_pain_conditions")

    charlson_total_na = df["CCI_Score"].isna().sum().sum()
    cci_summary = summarize_category_column(df, "CCI_Score")

    comorbidx_total_na = df["FCI_Score"].isna().sum().sum()
    fci_summary = summarize_integer_column(df, "FCI_Score")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as md_file:

        md_file.write("# Data Summary Report\n\n")

        md_file.write("## 1. Comorbidities Dataset Overview\n\n")
        md_file.write("- Total NA Count: {}\n\n".format(comorb_total_na))

        md_file.write("### Comorbidities Prevalence Rate\n\n")
        md_file.write(prevalence_df.to_markdown() + "\n\n")

        md_file.write("### chronic_pain_conditions Overview\n\n")
        md_file.write("- Total NA Count: {}\n\n".format(chronic_pain_conditions_total_na))

        md_file.write("### chronic_pain_conditions Summary\n\n")
        chronic_pain_conditions_summary_df = pd.DataFrame.from_dict(
            chronic_pain_conditions_summary, orient='index', columns=['Value']
        )
        md_file.write(chronic_pain_conditions_summary_df.to_markdown() + "\n\n")

        md_file.write("## 2. Charlson Index Dataset Overview\n\n")
        md_file.write("- Total NA Count: {}\n\n".format(charlson_total_na))

        md_file.write("### CCI_Score Summary\n\n")
        cci_summary_df = pd.DataFrame.from_dict(
            cci_summary, orient='index', columns=['Value']
        )
        md_file.write(cci_summary_df.to_markdown() + "\n\n")

        md_file.write("## 3. Comorbidity Index Dataset Overview\n\n")
        md_file.write("- Total NA Count: {}\n\n".format(comorbidx_total_na))

        md_file.write("### FCI_Score Summary\n\n")
        fci_summary_df = pd.DataFrame.from_dict(
            fci_summary, orient='index', columns=['Value']
        )
        md_file.write(fci_summary_df.to_markdown() + "\n\n")

    print(f"Markdown report has been generated and saved as '{output_file}'.")

generate_markdown_report(
    df_input,
    output_file="./outputs/data_summary/data_summary_report_2.md"
)
