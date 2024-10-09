
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
    #diabetes["house_family_person_id"] = diabetes.apply(lambda x: "_".join(x[["HHX", "FMX", "FPX"]].astype(int).astype(str)), 
    #                                                    axis=1)
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

    '''#create a household id column
    diabetes["household_id"] = diabetes["HHX"].astype(str)

    # create a family id column
    diabetes["family_id"] = diabetes["FMX"].astype(str)

    # create a person id column
    diabetes["person_id"] = diabetes["FPX"].astype(str)'''

    # create the diabetes column
    diabetes["diabetes"] = (diabetes["DIBEV1"] == 1).astype(int)
    # create coronary heart disease column
    diabetes["coronary_heart_disease"] = (diabetes["CHDEV"] == 1).astype(int)
    # create hypertension column
    diabetes["hypertension"] = (diabetes["HYPEV"] == 1).astype(int)
    # create heart_condition column
    diabetes["heart_condition"] = (diabetes["HRTEV"] == 1).astype(int)
    # create cancer column
    diabetes["cancer"] = (diabetes["CANEV"] == 1).astype(int)
    # create family_history_diabetes column
    diabetes["family_history_diabetes"] = (diabetes["DIBREL"] == 1).astype(int)
    # rename remaining relevant columns
    diabetes = diabetes.rename(columns={"AGE_P": "age",
                                      "SMKEV": "smoker",
                                      "SEX": "sex",
                                      "AWEIGHTP": "weight",
                                      "BMI": "bmi",
                                      "AHEIGHT": "height",
                                      "DBHVPAY" : "doctor_recommend_exercise",
                                      "MODTP": "moderate_physical_activity",
                                      "VIGTP": "vigorous_physical_activity",
                                      "ALC12MNO" : "alcohol_past_year",
                                      "HYPMDEV2": "high_blood_pressure_prescription",
                                      "REGION": "region",
                                      "R_MARITL": "marital_status"})

    # select just the relevant columns
    diabetes = diabetes[[#"household_id",
                        #"family_id",
                        #"person_id",
                        "house_family_person_id",
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
                        "high_blood_pressure_prescription",
                        "region",
                        "marital_status"]]
    return(diabetes)

