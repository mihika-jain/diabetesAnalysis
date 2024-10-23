
import pandas as pd
import numpy as np

def load_diabetes_data(path):
    # load in the original data
    diabetes_orig = pd.read_csv(path)

    # take just one person from each household
    diabetes = diabetes_orig.groupby("HHX") \
      .sample(1, random_state=24648765) \
      .reset_index() \
      .copy()
    # add an id column
    diabetes["id"] = np.arange(len(diabetes.index))
    # create the house_family_person_id column by joining together three ID columns

    # Determine the maximum number of digits for each column
    max_digits_HHX = diabetes["HHX"].astype(str).str.len().max()
    diabetes["FMX_padded"] = diabetes["FMX"].astype(str).str.zfill(2)
    diabetes["FPX_padded"] = diabetes["FPX"].astype(str).str.zfill(2)

    # Apply the padding and create the new ID
    diabetes["house_family_person_id"] = diabetes.apply(
        lambda x: int("".join([
            str(x["HHX"]).zfill(max_digits_HHX),
            x["FMX_padded"],
            x["FPX_padded"]
        ])),
        axis=1
    )

    # create the diabetes column
    diabetes["diabetes"] = diabetes["DIBEV1"]
    #cholesterol
    diabetes["cholesterol"] = diabetes["CHLEV"]
    #stroke
    diabetes["stroke"] = diabetes["STREV"]
    #taking insulin
    #diabetes["insulin"] = diabetes["INSLN1"]
    diabetes["blood_sugar"] = diabetes["APSBSCHK"]
    # create coronary heart disease column
    diabetes["coronary_heart_disease"] = diabetes["CHDEV"]
    # create hypertension column
    diabetes["hypertension"] = diabetes["HYPEV"]
    # create heart_condition column
    diabetes["heart_condition"] = diabetes["HRTEV"]
    # create cancer column
    diabetes["cancer"] = diabetes["CANEV"]
    # create family_history_diabetes column
    diabetes["family_history_diabetes"] = diabetes["DIBREL"]
    # rename remaining relevant columns
    diabetes = diabetes.rename(columns={"AGE_P": "age",
                                      "SMKEV": "smoker",
                                      "SEX": "sex",
                                      "AWEIGHTP": "weight",
                                      "BMI": "bmi",
                                      "AHEIGHT": "height",
                                      "DBHVPAY" : "doctor_recommend_exercise",
                                      "MODFREQW": "moderate_physical_activity",
                                      "VIGFREQW": "vigorous_physical_activity",
                                      "ALC12MNO" : "alcohol_past_year",
                                      "HYPMDEV2": "high_blood_pressure_prescription",
                                      "REGION": "region",
                                      "R_MARITL": "marital_status",
                                       "DBHVCLY" : "told_reduce_fat",
                                       "DBHVWLN" : "weight_loss_program",
                                       "ASISLEEP": "hours_sleep",
                                       "RACERPI2" : "race",
                                       "ANGEV": "angina_pectoris",
                                       "MIEV": "heart_attack",
                                       "KIDWKYR" : "kidney_weak",
                                       "FLA1AR" : "functional_limits",
                                       "PREGEVER" : "ever_pregnent",
                                       "ARTH1" : "arthritis",
                                       "AFLHCA17" : "depression",
                                       "ARX12_1" : "skip_medication",
                                       "AHCAFYR1" : "cannot_afford_medication"})

    # select just the relevant columns
    diabetes = diabetes[["house_family_person_id",
                        "diabetes",
                        "age",
                        "smoker",
                        "sex",
                        "coronary_heart_disease",
                        "weight",
                        "bmi",
                        "height",
                        "hypertension",
                        "heart_condition",
                        "cancer",
                        "family_history_diabetes",
                        "doctor_recommend_exercise",
                        "moderate_physical_activity",
                        "vigorous_physical_activity",
                        "alcohol_past_year",
                        "region",
                        "marital_status",
                        "stroke",
                        "told_reduce_fat",
                        "weight_loss_program",
                        "hours_sleep",
                        "blood_sugar",
                        "race",
                        "angina_pectoris",
                        "heart_attack",
                        "kidney_weak",
                        "functional_limits",
                        "arthritis",
                        "cannot_afford_medication"]]
    return(diabetes)

