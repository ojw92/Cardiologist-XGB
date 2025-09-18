#!/usr/bin/env python3

"""
# spark_etl.py

Cleans, normalizes, and splits the 2022 heart-health survey dataset in a
reproducible, scalable way using PySpark (Apache Spark). Outputs a cleaned 
dataset and stratified train/test splits that downstream XGBoost notebooks 
can consume.

Key Characteristics
---------------------------------
1) Determinism: stratified splits are reproducible with --seed (via sampleBy)
2) No leakage: test rows are removed from train set via left_anti join on a unique id
3) Schema stability: output column names and types are stable across runs
4) Idempotence: re-running with the same inputs overwrites outputs with same result
5) Feature parity: encodes and scales the same features used in the notebooks

Inputs
------
- CSV (header=true, inferSchema ok): heart_2022_no_nans.csv
  Expected key columns:
    - HadHeartAttack: "Yes"/"No" (mapped to 1/0)
    - Sex: "Male"/"Female" (mapped to 1/0)
    - GeneralHealth: ordinal labels {"Poor","Fair","Good","Very good","Excellent"}
    - RemovedTeeth: ordinal labels {"None of them","1 to 5","6 or more, but not all","All"}
    - AgeCategory: bucketed ages (e.g., "18-24",..."80 or older")
    - SmokerStatus: free-text variants → {"Never","Former","Some days","Every day"}
    - ECigaretteUsage: free-text variants → {"Never","Former","Some days","Every day"}
    - LastCheckupTime: simplified to {"Within past year","Within past 2 years","Over 2 years ago"}
    - TetanusLast10Tdap: simplified to {"No","Yes-Tdap","Yes-not Tdap","Yes-type unknown"}
    - PhysicalHealthDays, MentalHealthDays, SleepHours, HeightInMeters, WeightInKilograms, BMI
    - Optional: State (mapped to Region)
    - etc.

Outputs
-------
- {outdir}/heart_2022_clean.parquet
    * Cleaned columns with recodes (see "Feature Engineering") and scaled numeric columns:
      *_scaled for each of:
        PhysicalHealthDays, MentalHealthDays, SleepHours, HeightInMeters,
        WeightInKilograms, BMI, GeneralHealth, RemovedTeeth, AgeCategory,
        SmokerStatus_ord, ECigaretteUsage_ord, SmokerOrECig_ord
- {outdir}/train_df0.1.parquet, {outdir}/test_df0.1.parquet
    * Stratified by HadHeartAttack with fraction = --test_frac

Feature Engineering
-------------------
- HadHeartAttack: Yes/No → 1/0
- Sex: Female→0, Male→1
- GeneralHealth: {Poor:1, Fair:2, Good:3, Very good:4, Excellent:5}
- RemovedTeeth: {"None of them":1, "1 to 5":2, "6 or more, but not all":3, "All":4}
- AgeCategory: mapped to ordered integers (18–24→1, ..., 80 or older→13)
- SmokerStatus & ECigaretteUsage: normalized labels; ordinal map {"Never":1,"Former":2,"Some days":3,"Every day":4}
- SmokerOrECig_ord: max(SmokerStatus_ord, ECigaretteUsage_ord)
- LastCheckupTime/TetanusLast10Tdap: simplified label sets (see code)
- State→Region: US Census-style macro regions (optional if State exists)
- Scaling: Min–Max scaling via Spark ML MinMaxScaler → *_scaled columns

CLI (Command Line Interface)
---
python spark_etl.py \
  --input ../Data/heart_2022_no_nans.csv \
  --outdir ../Data \
  --test_frac 0.30 \
  --seed 25 \
  [--write_csv]
"""

import argparse
from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml.functions import vector_to_array

# ---- mappings (add more later) ----
GENERAL_HEALTH = {"Poor":1,"Fair":2,"Good":3,"Very good":4,"Excellent":5}
REMOVED_TEETH  = {"None of them":1,"1 to 5":2,"6 or more, but not all":3,"All":4}
AGE_LEVELS = ["18-24","25-29","30-34","35-39","40-44","45-49","50-54","55-59","60-64","65-69","70-74","75-79","80 or older"]
AGE_TO_ORD = {lvl:i+1 for i,lvl in enumerate(AGE_LEVELS)}
SMOKER_MAP = {"Never":1,"Former":2,"Some days":3,"Every day":4}

STATE_TO_REGION = {
    "Maine":"New_England","New Hampshire":"New_England","Vermont":"New_England",
    "Massachusetts":"New_England","Rhode Island":"New_England","Connecticut":"New_England",
    "New York":"Middle_Atlantic","New Jersey":"Middle_Atlantic","Pennsylvania":"Middle_Atlantic",
    "Ohio":"East_North_Central","Michigan":"East_North_Central","Indiana":"East_North_Central",
    "Wisconsin":"East_North_Central","Illinois":"East_North_Central",
    "Delaware":"South_Atlantic","Maryland":"South_Atlantic","District of Columbia":"South_Atlantic",
    "Washington, DC":"South_Atlantic","West Virginia":"South_Atlantic","Virginia":"South_Atlantic",
    "North Carolina":"South_Atlantic","South Carolina":"South_Atlantic","Georgia":"South_Atlantic","Florida":"South_Atlantic",
    "Kentucky":"East_South_Central","Tennessee":"East_South_Central","Alabama":"East_South_Central","Mississippi":"East_South_Central",
    "Arkansas":"West_South_Central","Louisiana":"West_South_Central","Oklahoma":"West_South_Central","Texas":"West_South_Central",
    "Minnesota":"West_North_Central","Iowa":"West_North_Central","Missouri":"West_North_Central","North Dakota":"West_North_Central",
    "South Dakota":"West_North_Central","Nebraska":"West_North_Central","Kansas":"West_North_Central",
    "New Mexico":"Mountain","Arizona":"Mountain","Colorado":"Mountain","Utah":"Mountain","Nevada":"Mountain","Wyoming":"Mountain","Idaho":"Mountain","Montana":"Mountain",
    "Washington":"Pacific","Oregon":"Pacific","California":"Pacific","Alaska":"Pacific","Hawaii":"Pacific"
}

# Columns
    # State; Sex; GeneralHealth; PhysicalHealthDays; MentalHealthDays; LastCheckupTime; PhysicalActivities; SleepHours; RemovedTeeth; HadHeartAttack; HadAngina;
    # HadStroke; HadAsthma; HadSkinCancer; HadCOPD; HadDepressiveDisorder; HadKidneyDisease; HadArthritis; HadDiabetes DeafOrHardOfHearing; BlindOrVisionDifficulty;
    # DifficultyConcentrating; DifficultyWalking; DifficultyDressingBathing; DifficultyErrands; SmokerStatus; ECigaretteUsage; ChestScan; RaceEthnicityCategory;
    # AgeCategory; HeightInMeters; WeightInKilograms; BMI; AlcoholDrinkers; HIVTesting; FluVaxLast12; PneumoVaxEver; TetanusLast10Tdap; HighRiskLastYear; CovidPos;

def main():
    # Create parser for command-line options with help messages
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to heart_2022_no_nans.csv")
    ap.add_argument("--outdir", required=True, help="Output directory (e.g., ../Data)")
    ap.add_argument("--test_frac", type=float, default=0.3)     # test set 30%
    ap.add_argument("--seed", type=int, default=25)
    ap.add_argument("--write_csv", action="store_true", help="Also write CSV alongside Parquet")
    args = ap.parse_args()

    # Create Spark session
    spark = (SparkSession.builder
             .appName("HeartDisease-Spark-ETL")
             .getOrCreate())

    # Read csv file by inferring schema; define explicit StructType if preferred
    df = (spark.read
          .option("header", True)
          .option("inferSchema", True)
          .csv(args.input))

    # Map binary features to 1 and 0
    yes_no_cols = ["HadHeartAttack", "HadAngina", "HadStroke", "HadAsthma", "HadCOPD", "HadDepressiveDisorder", "HadKidneyDisease",
                    "HadArthritis", "DeafOrHardOfHearing", "BlindOrVisionDifficulty", "DifficultyConcentrating", "DifficultyWalking",
                    "DifficultyDressingBathing", "DifficultyErrands", "ChestScan", "AlcoholDrinkers", "HIVTesting", "FluVaxLast12",
                    "PneumoVaxEver", "TetanusLast10Tdap", "HighRiskLastYear"]
    for c in yes_no_cols:
        df = df.withColumn(c, F.when(F.col(c)=="Yes", F.lit(1)).when(F.col(c)=="No", F.lit(0)).otherwise(F.col(c)))

    df = df.withColumn("Sex", F.when(F.col("Sex")=="Female", F.lit(0)).when(F.col("Sex")=="Male", F.lit(1)).otherwise(F.col("Sex")))

    # GeneralHealth/RemovedTeeth/AgeCategory value conversion to integer
    def map_by_dict(col, m):
        return F.coalesce(F.create_map([F.lit(x) for kv in m.items() for x in kv])[F.col(col)], F.col(col))

    df = df.withColumn("GeneralHealth", map_by_dict("GeneralHealth", GENERAL_HEALTH).cast("int"))
    df = df.withColumn("RemovedTeeth",  map_by_dict("RemovedTeeth",  REMOVED_TEETH).cast("int"))
    df = df.withColumn("AgeCategory",   map_by_dict("AgeCategory",   AGE_TO_ORD).cast("int"))

    # LastCheckupTime simplification
    df = df.withColumn("LastCheckupTime",
        F.when(F.col("LastCheckupTime").contains("anytime less than 12 months"), F.lit("Within past year"))
         .when(F.col("LastCheckupTime").contains("less than 2 years"), F.lit("Within past 2 years"))
         .when((F.col("LastCheckupTime")=="5 or more years ago") | (F.col("LastCheckupTime").contains("less than 5 years")), F.lit("Over 2 years ago"))
         .otherwise(F.col("LastCheckupTime"))
    )

    # Tetanus wording
    df = df.withColumn("TetanusLast10Tdap",
        F.when(F.col("TetanusLast10Tdap").startswith("No, did not receive"), F.lit("No"))
         .when(F.col("TetanusLast10Tdap")=="Yes, received Tdap", F.lit("Yes-Tdap"))
         .when(F.col("TetanusLast10Tdap")=="Yes, received tetanus shot, but not Tdap", F.lit("Yes-not Tdap"))
         .when(F.col("TetanusLast10Tdap").startswith("Yes, received tetanus shot but not sure"), F.lit("Yes-type unknown"))
         .otherwise(F.col("TetanusLast10Tdap"))
    )

    # Smoker / eCig simplify + ordinal
    df = df.withColumn("SmokerStatus",
        F.when(F.col("SmokerStatus")=="Never smoked", F.lit("Never"))
         .when(F.col("SmokerStatus")=="Former smoker", F.lit("Former"))
         .when(F.col("SmokerStatus").contains("some days"), F.lit("Some days"))
         .when(F.col("SmokerStatus").contains("every day"), F.lit("Every day"))
         .otherwise(F.col("SmokerStatus"))
    )
    df = df.withColumn("ECigaretteUsage",
        F.when(F.col("ECigaretteUsage")=="Never used e-cigarettes in my entire life", F.lit("Never"))
         .when(F.col("ECigaretteUsage")=="Not at all (right now)", F.lit("Former"))
         .when(F.col("ECigaretteUsage")=="Use them some days", F.lit("Some days"))
         .when(F.col("ECigaretteUsage")=="Use them every day", F.lit("Every day"))
         .otherwise(F.col("ECigaretteUsage"))
    )
    df = df.withColumn("SmokerStatus_ord", map_by_dict("SmokerStatus", SMOKER_MAP).cast("int"))
    df = df.withColumn("ECigaretteUsage_ord", map_by_dict("ECigaretteUsage", SMOKER_MAP).cast("int"))
    df = df.withColumn("SmokerOrECig_ord", F.greatest("SmokerStatus_ord","ECigaretteUsage_ord"))     # returns the greatest value of the list of column names, skipping null values

    # State conversion to Region
    region_map = F.create_map([F.lit(x) for kv in STATE_TO_REGION.items() for x in kv])
    if "State" in df.columns:
        df = df.withColumn("Region", region_map[F.col("State")])

    # MinMax scaling (same column names: *_scaled)
    cnts_cols = [
        "PhysicalHealthDays","MentalHealthDays","SleepHours","HeightInMeters",
        "WeightInKilograms","BMI","GeneralHealth","RemovedTeeth","AgeCategory",
        "SmokerStatus_ord","ECigaretteUsage_ord","SmokerOrECig_ord"
    ]
    for c in cnts_cols:
        if c not in df.columns:
            raise ValueError(f"Expected numeric column {c} not found in input schema.")

    vec_in, vec_out = "cnts_vec", "cnts_vec_scaled"
    assembler = VectorAssembler(inputCols=cnts_cols, outputCol=vec_in)
    df = assembler.transform(df)
    scaler = MinMaxScaler(inputCol=vec_in, outputCol=vec_out)
    scaler_model = scaler.fit(df)
    df = scaler_model.transform(df)
    arr = vector_to_array(F.col(vec_out))
    for i, c in enumerate(cnts_cols):
        df = df.withColumn(f"{c}_scaled", arr[i])

    # Reproducible stratified split on HadHeartAttack
    if "HadHeartAttack" not in df.columns:
        raise ValueError("Target column 'HadHeartAttack' not found.")
    df = df.withColumn("row_id", F.monotonically_increasing_id())

    # Make exact fractions map like {0: test_frac, 1: test_frac}
    label_vals = [r[0] for r in df.select("HadHeartAttack").distinct().collect()]
    fractions = {int(v): args.test_frac for v in label_vals}
    test = df.stat.sampleBy("HadHeartAttack", fractions, seed=args.seed)
    train = df.join(test.select("row_id"), on="row_id", how="left_anti")

    # Write outputs
    clean_path = f"{args.outdir}/heart_2022_clean.parquet"
    train_path = f"{args.outdir}/train_df0.1.parquet"
    test_path  = f"{args.outdir}/test_df0.1.parquet"

    # Persist clean (pre-split) for EDA reuse
    (df.drop(vec_in, vec_out).write.mode("overwrite").parquet(clean_path))
    train.drop(vec_in, vec_out).write.mode("overwrite").parquet(train_path)
    test.drop(vec_in, vec_out).write.mode("overwrite").parquet(test_path)

    # Write cleaned, train and test sets
    if args.write_csv:
        df.drop(vec_in, vec_out).coalesce(1).write.mode("overwrite").option("header", True).csv(clean_path.replace(".parquet","_csv"))
        train.drop(vec_in, vec_out).coalesce(1).write.mode("overwrite").option("header", True).csv(train_path.replace(".parquet","_csv"))
        test.drop(vec_in, vec_out).coalesce(1).write.mode("overwrite").option("header", True).csv(test_path.replace(".parquet","_csv"))

    print(f"✅ Wrote:\n  {clean_path}\n  {train_path}\n  {test_path}")

if __name__ == "__main__":
    main()
