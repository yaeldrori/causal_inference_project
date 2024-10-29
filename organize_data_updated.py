import pandas as pd
import glob
import re
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import glob
import re
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import os
import pickle


class AbnormalChecker:
    @staticmethod
    def tag_rhythm(rhythm):
        rhythm = rhythm.strip().upper()
        sinus_rhythms = ['SR', 'SA', 'SB', 'ST']
        if any(sinus in rhythm for sinus in sinus_rhythms):
            return 0  # Sinus rhythms
        elif 'AF' in rhythm or 'A FLUT' in rhythm or 'ATRIAL FLUTTER' in rhythm:
            return 1  # Atrial fibrillation or flutter
        else:
            return 1  # Any other rhythm

    @staticmethod
    def check_abnormal_rhythm(row, proc_date):
        rhythm_list = row['Value']  # Assuming the column with rhythm values is 'values_rhythm'
        rhythm_times = row['Date']  # Assuming you have timestamps for rhythm checks

        first_rhythm_value = 0
        last_rhythm_value = 0

        if proc_date != 0.0 and proc_date != 0:
            proc_date = pd.to_datetime(proc_date)

            # Check all rhythms before the procedure date
            rhythms_before = [
                rhythm for rhythm, time in zip(rhythm_list, rhythm_times)
                if pd.to_datetime(time) < proc_date
            ]
            if rhythms_before:
                # If any rhythm before the procedure is abnormal, set to 1; otherwise, -1
                first_rhythm_value = (
                    1 if any(AbnormalChecker.tag_rhythm(r) for r in rhythms_before) else -1
                )
            else:
                first_rhythm_value = 0  # No data before the procedure

            # Check all rhythms after the procedure date
            rhythms_after = [
                rhythm for rhythm, time in zip(rhythm_list, rhythm_times)
                if pd.to_datetime(time) > proc_date
            ]
            if rhythms_after:
                # If any rhythm after the procedure is abnormal, set to 1; otherwise, -1
                last_rhythm_value = (
                    1 if any(AbnormalChecker.tag_rhythm(r) for r in rhythms_after) else -1
                )
            else:
                last_rhythm_value = 0  # No data after the procedure
        else:
            # Use the first and last rhythm value if proc_date is 0
            if len(rhythm_list.values)>0:
                first_rhythm_value = (
                    1 if AbnormalChecker.tag_rhythm(rhythm_list[0]) else -1
                )
                last_rhythm_value = (
                    1 if AbnormalChecker.tag_rhythm(rhythm_list[len(rhythm_list.values)-1]) else -1
                )
            else:
                # No rhythm data available
                first_rhythm_value = 0
                last_rhythm_value = 0

        return first_rhythm_value, last_rhythm_value


    @staticmethod
    def is_abnormal_blood(label, value):
        boundaries = {        'Hemoglobin A1c': {'lower_bound': 4.0, 'upper_bound': 5.6, 'units': '%'},
        '24 hr Calcium': {'lower_bound': 2.1, 'upper_bound': 2.6, 'units': 'mmol/L'},
        '24 hr Creatinine': {'lower_bound': 45, 'upper_bound': 110, 'units': 'µmol/L'},
        '25-OH Vitamin D': {'lower_bound': 30, 'upper_bound': 100, 'units': 'ng/mL'},
        'Alanine Aminotransferase (ALT)': {'lower_bound': 7, 'upper_bound': 56, 'units': 'U/L'},
        'Albumin': {'lower_bound': 3.5, 'upper_bound': 5.0, 'units': 'g/dL'},
        'Alkaline Phosphatase': {'lower_bound': 44, 'upper_bound': 147, 'units': 'U/L'},
        'Amylase': {'lower_bound': 23, 'upper_bound': 85, 'units': 'U/L'},
        'Anion Gap': {'lower_bound': 8, 'upper_bound': 16, 'units': 'mmol/L'},
        'Aspartate Aminotransferase (AST)': {'lower_bound': 10, 'upper_bound': 40, 'units': 'U/L'},
        'Bicarbonate': {'lower_bound': 22, 'upper_bound': 29, 'units': 'mmol/L'},
        'Bilirubin': {'lower_bound': 0.1, 'upper_bound': 1.2, 'units': 'mg/dL'},
        'Calcium': {'lower_bound': 8.5, 'upper_bound': 10.5, 'units': 'mg/dL'},
        'Chloride': {'lower_bound': 96, 'upper_bound': 106, 'units': 'mmol/L'},
        'Cholesterol': {'lower_bound': 0, 'upper_bound': 200, 'units': 'mg/dL'},
        'Creatine Kinase (CK)': {'lower_bound': 20, 'upper_bound': 200, 'units': 'U/L'},
        'Creatinine': {'lower_bound': 0.6, 'upper_bound': 1.3, 'units': 'mg/dL'},
        'Ferritin': {'lower_bound': 30, 'upper_bound': 400, 'units': 'ng/mL'},
        'Folate': {'lower_bound': 3.1, 'upper_bound': 17.5, 'units': 'ng/mL'},
        'Free Calcium': {'lower_bound': 4.5, 'upper_bound': 5.6, 'units': 'mg/dL'},
        'Glucose': {'lower_bound': 70, 'upper_bound': 100, 'units': 'mg/dL'},
        'Hemoglobin': {'lower_bound': 13.8, 'upper_bound': 17.2, 'units': 'g/dL'},
        'Iron': {'lower_bound': 60, 'upper_bound': 170, 'units': 'µg/dL'},
        'Lactate': {'lower_bound': 0.5, 'upper_bound': 2.2, 'units': 'mmol/L'},
        'Magnesium': {'lower_bound': 1.7, 'upper_bound': 2.2, 'units': 'mg/dL'},
        'Phosphate': {'lower_bound': 2.5, 'upper_bound': 4.5, 'units': 'mg/dL'},
        'Potassium': {'lower_bound': 3.5, 'upper_bound': 5.0, 'units': 'mmol/L'},
        'Protein': {'lower_bound': 6.0, 'upper_bound': 8.3, 'units': 'g/dL'},
        'Sodium': {'lower_bound': 135, 'upper_bound': 145, 'units': 'mmol/L'},
        'Thyroid Stimulating Hormone (TSH)': {'lower_bound': 0.4, 'upper_bound': 4.0, 'units': 'mIU/L'},
        'Triglycerides': {'lower_bound': 0, 'upper_bound': 150, 'units': 'mg/dL'},
        'Urea Nitrogen (BUN)': {'lower_bound': 6, 'upper_bound': 20, 'units': 'mg/dL'},
        'Uric Acid': {'lower_bound': 3.5, 'upper_bound': 7.2, 'units': 'mg/dL'},
        'Vitamin B12': {'lower_bound': 200, 'upper_bound': 900, 'units': 'pg/mL'},
        'WBC Count': {'lower_bound': 4.0, 'upper_bound': 11.0, 'units': '10^3/µL'},
        'd_dimer': {'lower_bound': 0, 'upper_bound': 0.5, 'units': 'µg/mL FEU'},
        # D-dimer often varies, this is a common cutoff
        'fibrinogen': {'lower_bound': 200, 'upper_bound': 400, 'units': 'mg/dL'},
        'inr': {'lower_bound': 0.8, 'upper_bound': 1.2, 'units': ''},  # INR values can vary based on therapeutic goals
        'pt': {'lower_bound': 11, 'upper_bound': 13.5, 'units': 'seconds'},
        'ptt': {'lower_bound': 25, 'upper_bound': 35, 'units': 'seconds'},  # PTT can also vary depending on context
        'thrombin': {'lower_bound': 14, 'upper_bound': 21, 'units': 'seconds'},
        }

        if label in boundaries:
            bounds = boundaries[label]
            if pd.notna(value):
                if value < bounds['lower_bound'] or value > bounds['upper_bound']:
                    return 1  # Abnormal
        return 0  # Normal



    @staticmethod
    def check_abnormal_hosp_values(row, proc_date):
        hosp_labels = list(row['Test Name'].values)  # Blood test names
        hosp_values = list(row['Value'].values)  # Corresponding test values
        blood_test_times = list(row['Date'].values)  # Timestamps for blood tests

        first_value_status = 0
        last_value_status = 0

        if proc_date != 0.0 and proc_date != 0:
            proc_date = pd.to_datetime(proc_date)

            # Check all blood tests before the procedure date
            values_before = [
                (label, value) for label, value, time in zip(hosp_labels, hosp_values, blood_test_times)
                if pd.to_datetime(time) < proc_date
            ]
            if values_before:
                # If any value before is abnormal, set to 1; otherwise, -1
                first_value_status = (
                    1 if any(AbnormalChecker.is_abnormal_blood(label, float(value))
                             for label, value in values_before) else -1
                )
            else:
                first_value_status = 0  # No data before procedure

            # Check all blood tests after the procedure date
            values_after = [
                (label, value) for label, value, time in zip(hosp_labels, hosp_values, blood_test_times)
                if pd.to_datetime(time) > proc_date
            ]
            if values_after:
                # If any value after is abnormal, set to 1; otherwise, -1
                last_value_status = (
                    1 if any(AbnormalChecker.is_abnormal_blood(label, float(value))
                             for label, value in values_after) else -1
                )
            else:
                last_value_status = 0  # No data after procedure

        else:
            # If proc_date is 0, use the first and last blood test values
            if len(hosp_labels) > 0:
                # Check the first blood test value
                first_value_status = (
                    1 if AbnormalChecker.is_abnormal_blood(hosp_labels[0], float(hosp_values[0])) else -1
                )
                # Check the last blood test value
                last_value_status = (
                    1 if AbnormalChecker.is_abnormal_blood(hosp_labels[-1], float(hosp_values[-1])) else -1
                )
            else:
                # No data available
                first_value_status = 0
                last_value_status = 0

        return first_value_status, last_value_status

    @staticmethod
    def check_abnormal_coagulation_values(row, proc_date):
        coag_labels = list(row['Test Name'].values)
        coag_values = list(row['Value'].values)
        coag_test_times = list(row['Date'].values)

        first_value_status = 0
        last_value_status = 0

        if proc_date != 0.0 and proc_date != 0:
            proc_date = pd.to_datetime(proc_date)

            # Check all coagulation tests before the procedure date
            values_before = [
                (label, value) for label, value, time in zip(coag_labels, coag_values, coag_test_times)
                if pd.to_datetime(time) < proc_date
            ]
            if values_before:
                # If any value before is abnormal, set to 1; otherwise, -1
                first_value_status = (
                    1 if any(AbnormalChecker.is_abnormal_blood(label, float(value))
                             for label, value in values_before) else -1
                )
            else:
                first_value_status = 0  # No data before procedure

            # Check all coagulation tests after the procedure date
            values_after = [
                (label, value) for label, value, time in zip(coag_labels, coag_values, coag_test_times)
                if pd.to_datetime(time) > proc_date
            ]
            if values_after:
                # If any value after is abnormal, set to 1; otherwise, -1
                last_value_status = (
                    1 if any(AbnormalChecker.is_abnormal_blood(label, float(value))
                             for label, value in values_after) else -1
                )
            else:
                last_value_status = 0  # No data after procedure

        else:
            # Use the first and last coagulation test values if proc_date is 0
            if len(coag_labels) > 0:
                # Check the first coagulation test value
                first_value_status = (
                    1 if AbnormalChecker.is_abnormal_blood(coag_labels[0], float(coag_values[0])) else -1
                )
                # Check the last coagulation test value
                last_value_status = (
                    1 if AbnormalChecker.is_abnormal_blood(coag_labels[-1], float(coag_values[-1])) else -1
                )
            else:
                # No data available
                first_value_status = 0
                last_value_status = 0

        return first_value_status, last_value_status

    @staticmethod
    def process_diagnosis(row):
        """
        Adds one-hot encoded columns for each unique value in the list of codes for each patient.

        Parameters:
            df (pd.DataFrame): The original DataFrame containing patient data.
            column_name (str): The name of the column containing the list of diagnosis codes.

        Returns:
            pd.DataFrame: The original DataFrame with additional one-hot encoded columns.
        """

        cardiac_related_diagnoses = [
            "Acute and subacute bacterial endocarditis",
            "Acute combined systolic and diastolic heart failure",
            "Acute diastolic heart failure",
            "Acute idiopathic pericarditis",
            "Acute myocardial infarction of inferolateral wall",
            "Acute myocardial infarction of other lateral wall",
            "Acute myocardial infarction of other specified sites",
            "Acute myocardial infarction of unspecified site",
            "Acute on chronic combined systolic and diastolic heart failure",
            "Acute on chronic diastolic heart failure",
            "Acute on chronic systolic heart failure",
            "Acute pericarditis",
            "Acute systolic heart failure",
            "Aortic aneurysm of unspecified site without mention of rupture",
            "Aortic ectasia",
            "Aortic valve disorders",
            "Aortocoronary bypass status",
            "Atrial fibrillation",
            "Atrial flutter",
            "Atrioventricular block",
            "Cardiac arrest",
            "Cardiac complications",
            "Cardiac dysrhythmia",
            "Cardiac pacemaker in situ",
            "Cardiac tamponade",
            "Cardiogenic shock",
            "Cardiomegaly",
            "Cardiomyopathy in other diseases classified elsewhere",
            "Congestive heart failure",
            "Constrictive pericarditis",
            "Coronary atherosclerosis of autologous vein bypass graft",
            "Coronary atherosclerosis of native coronary artery",
            "Coronary atherosclerosis of unspecified type of vessel",
            "Coronary vasodilators causing adverse effects in therapeutic use",
            "Diastolic heart failure",
            "Hypertensive heart and chronic kidney disease",
            "Hypertrophic obstructive cardiomyopathy",
            "Mitral stenosis",
            "Mitral stenosis with insufficiency",
            "Mitral valve disorders",
            "Mitral valve insufficiency and aortic valve insufficiency",
            "Mitral valve insufficiency and aortic valve stenosis",
            "Mitral valve stenosis and aortic valve insufficiency",
            "Mitral valve stenosis and aortic valve stenosis",
            "Mobitz (type) II atrioventricular block",
            "Primary pulmonary hypertension",
            "Pulmonary valve disorders",
            "Rheumatic aortic insufficiency",
            "Rheumatic aortic stenosis",
            "Rheumatic heart disease",
            "Rheumatic heart failure (congestive)",
            "Rheumatic mitral insufficiency",
            "Right bundle branch block",
            "Right bundle branch block and left anterior fascicular block",
            "Sinoatrial node dysfunction",
            "Systolic heart failure",
            "Takotsubo syndrome",
            "Tricuspid valve disorders",
            "Ventricular fibrillation",
            "Ventricular flutter"
        ]

        test_names = row['Test Name'].values
        test_values = row['Value'].values
        test_dates = row['Date'].values

        # Initialize flags
        cardiac_procedure = 0
        cardioversion = 0
        cardioversion_date = None

        # Check for cardiac-related procedures
        cardiac_related = any(test_name in cardiac_related_diagnoses for test_name in test_names)
        cardiac_diag = 1 if cardiac_related else 0


        return cardiac_diag


    @staticmethod
    def process_procedures(row):
        """
        Process the procedures for each patient and add relevant columns to the DataFrame.

        Parameters:
            df (pd.DataFrame): The original DataFrame containing patient data.
            column_name (str): The name of the column containing the string of procedures.

        Returns:
            pd.DataFrame: The DataFrame with added columns for 'cardioversion', 'procedures'
        """

        cardiac_or_invasive_procedures = [
            "(Aorto)coronary bypass of two coronary arteries",
            "Ambulatory cardiac monitoring",
            "Angiocardiography",
            "Aortography",
            "Arterial catheterization",
            "Biopsy of heart",
            "Cardiac mapping",
            "Cardiopulmonary resuscitation",
            "Cardiotomy",
            "Cardiovascular stress test using treadmill",
            "Catheter based invasive electrophysiologic testing",
            "Coronary arteriography using a single catheter",
            "Coronary arteriography using two catheters",
            "Implantation of automatic cardioverter/defibrillator",
            "Implantation of cardiac resynchronization defibrillator",
            "Implantation of cardiac resynchronization pacemaker",
            "Implantation or replacement of automatic cardioverter/defibrillator",
            "Insertion of drug-eluting coronary artery stent(s)",
            "Left heart cardiac catheterization",
            "Open and other replacement of aortic valve with tissue graft",
            "Open and other replacement of mitral valve",
            "Open heart valvuloplasty of tricuspid valve without replacement",
            "Percutaneous transluminal coronary angioplasty [PTCA]",
            "Pericardiocentesis",
            "Pericardiotomy",
            "Repair of atrial septal defect with tissue graft",
            "Right heart cardiac catheterization",
            "Single internal mammary-coronary artery bypass",
            "Thoracentesis"
        ]

        test_names = row['Test Name'].values
        test_values = row['Value'].values
        test_dates = row['Date'].values



        # Initialize flags
        cardiac_procedure = 0
        cardioversion = 0
        cardioversion_date = 0

        # Check for cardiac-related procedures
        cardiac_related = any(test_name in cardiac_or_invasive_procedures for test_name in test_names)
        cardiac_procedure = 1 if cardiac_related else 0

        # Check for cardioversion procedure
        for i, test_name in enumerate(test_names):
            if 'version' in test_name.lower():
                cardioversion = 1
                cardioversion_date = pd.to_datetime(test_dates[i]).date()
                break  # Stop after finding the first occurrence of cardioversion

        return cardiac_procedure,cardioversion,cardioversion_date


class DataProcessor:
    @staticmethod
    def calculate_months_passed(row) -> int:
        """
            Calculates the number of months passed between two dates in a DataFrame row.
        If there is no 'dod' (Date of Death), returns 100.

        Parameters:
            row (pd.Series): The DataFrame row containing the date strings.

        Returns:
            int: The number of months passed between the two dates.
        """
        # Extract and parse the date strings

        if isinstance(row['admittime'], str):
            admit_date = datetime.strptime(row['admittime'].split('T')[0], '%Y-%m-%d')
        else:
            admit_date = row['admittime']

        # Check if 'dod' is missing
        if pd.isna(row['dod']) or row['dod'] == '':
            return 1000

        # Extract and parse the date of death
        if isinstance(row['dod'], str):
            dod = datetime.strptime(row['dod'], '%m/%d/%Y')
        else:
            dod = row['dod']

        # Calculate the difference in months
        difference = relativedelta(dod, admit_date)
        months_passed = difference.years * 12 + difference.months

        return months_passed

    @staticmethod
    def remove_duplicate_columns(df):
        # Find duplicate columns
        duplicate_columns = df.columns[df.columns.duplicated()].tolist()

        if duplicate_columns:
            print(f"Duplicate columns found: {duplicate_columns}")

            # Check if 'patient_id' columns are among the duplicate columns
            patient_id_columns = [col for col in duplicate_columns if 'patient_id' in col]

            # If there are multiple 'patient_id' columns, check if they all have the same values
            if patient_id_columns:
                # Take the first 'patient_id' column's values
                first_patient_id_column = df[patient_id_columns[0]]

                # Compare the values of the other 'patient_id' columns to the first one
                for col in patient_id_columns[1:]:
                    if not first_patient_id_column.equals(df[col]):
                        print(f"Warning: 'patient_id' columns '{patient_id_columns[0]}' and '{col}' do not match.")
                        return df  # Return the original DataFrame if they don't match

                print("All 'patient_id' columns have the same values. Proceeding with removal of duplicate columns.")

        # Remove duplicate columns (this will remove the extra 'patient_id' columns if their values match)
        df = df.loc[:, ~df.columns.duplicated()]

        return df

    @staticmethod
    def process_test_data_columns(df,new_df, columns):
        """
        Process several columns in a DataFrame that contain test data strings,
        and replace them with new DataFrames in each cell.

        Parameters:
        df (pd.DataFrame): The original DataFrame.
        columns (list): List of column names to process.

        Returns:
        pd.DataFrame: The updated DataFrame with processed DataFrames in specific cells.
        """

        # Iterate over the specified columns to process
        for col in columns:
            # Only process rows where the column is a string
            for idx, row in df.iterrows():
                if isinstance(row[col], str):
                    processed_results_df = DataProcessor.process_row_to_df(row[col], col)
                    # new_df.at[idx, col] = processed_results_df  # Insert the DataFrame back into the original cell

                    if col == 'procedures':
                        # procedures - add if cardiac relate procedure
                        # cardioversion col - 1 if had cardioversion, 0 if not
                        # procedures col - 0 if has no procedures, 1 if has cardioversion, 2 if had additional procedures
                        cardiac_procedure,cardioversion,cardioversion_date = AbnormalChecker.process_procedures(processed_results_df)
                        new_df.at[idx, 'cardiac_procedure'] = cardiac_procedure
                        new_df.at[idx, 'cardioversion_procedure'] = cardioversion
                        new_df.at[idx, 'cardioversion_procedure_date'] = cardioversion_date

                    if col == 'other_diagnoses':
                    # other cardiac diagnosis
                        cardiac_diag = AbnormalChecker.process_diagnosis(processed_results_df)
                        new_df.at[idx, 'cardiac_diagnosis'] = cardiac_diag

                    if col == 'hosp_blood':
                        first_quarter_abnormal, last_quarter_abnormal = AbnormalChecker.check_abnormal_hosp_values(processed_results_df,
                                                                                                   new_df.iloc[idx]['cardioversion_procedure_date'])
                        new_df.at[idx,'blood_admit'] = first_quarter_abnormal
                        new_df.at[idx, 'blood_discharge'] = last_quarter_abnormal

                    if col == 'coagulation':
                        first_quarter_abnormal, last_quarter_abnormal = AbnormalChecker.check_abnormal_coagulation_values(
                            processed_results_df,new_df.iloc[idx][ 'cardioversion_procedure_date'])
                        new_df.at[idx, 'coag_admit'] = first_quarter_abnormal
                        new_df.at[idx, 'coag_discharge'] = last_quarter_abnormal

                    if col == 'rhythm':
                        first_quarter_abnormal, last_quarter_abnormal = AbnormalChecker.check_abnormal_rhythm(processed_results_df,
                                                                                                              new_df.iloc[idx]['cardioversion_procedure_date'])

                        new_df.at[idx, 'rhythm_admit'] = first_quarter_abnormal
                        new_df.at[idx, 'rhythm_discharge']= last_quarter_abnormal

                else:
                    if col == 'procedures':
                        new_df.at[idx, 'cardiac_procedure'] = 0
                        new_df.at[idx, 'cardioversion_procedure'] = 0
                        new_df.at[idx, 'cardioversion_procedure_date'] = 0

                    if col == 'other_diagnoses':
                        new_df.at[idx, 'cardiac_diagnosis'] = 0

                    if col == 'hosp_blood':
                        new_df.at[idx,'blood_admit'] = 0
                        new_df.at[idx, 'blood_discharge'] = 0

                    if col == 'coagulation':
                        new_df.at[idx, 'coag_admit'] = 0
                        new_df.at[idx, 'coag_discharge'] = 0

                    if col == 'rhythm':
                        new_df.at[idx, 'rhythm_admit'] = 0
                        new_df.at[idx, 'rhythm_discharge']= 0
        return new_df

    @staticmethod
    def process_column_diagnosis(value):
        # Check if the value is a string and contains the expected format
        if isinstance(value, str):
            # Split the string into date, time, and name components
            try:
                # Split first by the comma to separate the date-time and the name
                date_time, name = value.split(',', 1)
                date_time = date_time.strip()  # Remove any surrounding spaces from date_time
                name = name.strip()  # Remove any surrounding spaces from name

                # Further split date_time into date and time
                date, time = date_time.split(' ', 1)
                return date.strip(), time.strip(), name.strip()
            except ValueError:
                # Return empty values if there's a format error
                return None, None, None
        # If the value is NaN, return None for each component
        elif pd.isna(value):
            return None, None, None
        # If the value is already a list, return as is (in case you encounter lists)
        elif isinstance(value, list):
            return value
        # Handle any other unexpected types
        else:
            return None, None, None

    @staticmethod
    def process_row_to_df(row, col):
        """
        Process a single row and return it as a DataFrame with Date, Time, Test Name, and Value.

        Parameters:
        row (str): The string to process.
        col (str): The name of the column being processed.

        Returns:
        pd.DataFrame: A DataFrame with 'Date', 'Time', 'Test Name', and 'Value' columns.
        """
        try:
            # Split the test results based on the specific column type
            if col in ['procedures']:
                #test_results = re.split(r'(?<=\))\s*,\s*|(?<=[a-zA-Z])\s*,\s*', row)
                pattern = r'(\d{4}-\d{2}-\d{2}:[^,]+(?:, [^,\d]+)*)'
                # Extract all matching patterns from the row
                test_results = re.findall(pattern, row)
            elif col in ['rhythm', 'other_diagnoses']:
                test_results = re.split(r'(?<=\))\s*,\s*|(?<=[a-zA-Z])\s*,\s*', row)

            else:
                test_results = re.split(r'(?<=\d), ', row)

            # Collect processed results in a list of dictionaries
            processed_results = []

            # Process each test result
            for result in test_results:
                if re.search(r'\d', result):  # Check if the result contains any number
                    try:
                        # Split into datetime and test-value parts
                        datetime_part, test_value_part = re.split(r'(?<!\d)(?=[a-zA-Z%#&])', result, maxsplit=1)

                        # Further split datetime into date and time
                        data_time_split = datetime_part.split(' ')
                        if col == 'procedures':
                            date = datetime_part.rstrip(':;')
                            time = None
                        else:
                            date = data_time_split[0]
                            time = data_time_split[1].rstrip(':')

                        if col == 'other_diagnoses' or col == 'procedures':
                            value = None
                            test_name = test_value_part
                        else:
                            # Split the test-value part into test name and value
                            test_name, value = test_value_part.split('=')

                        # Append the result to the processed results list
                        processed_results.append({
                            'Date': date,
                            'Time': time,
                            'Test Name': test_name,
                            'Value': value
                        })
                    except Exception as e:
                        # Skip results that can't be processed correctly
                        continue

            # Convert the list of processed results into a DataFrame if not empty
            if processed_results:
                results_df = pd.DataFrame(processed_results)
                return results_df
            else:
                return np.nan

        except Exception as e:
            # Return NaN if there's an error in processing
            return np.nan


class FileLoader:
    @staticmethod
    def load_files(file_pattern):
        return glob.glob(file_pattern)

    @staticmethod
    def read_excel_files(file_list):
        dfs = [pd.read_excel(file, header=1) for file in file_list]
        return pd.concat(dfs, axis=0)


output_path = r'C:\Users\yaelp\OneDrive - Technion\causal_inference_project\Combined_data.pkl'
if os.path.exists(output_path):
    with open(output_path, 'rb') as file:
        combined_df = pickle.load(file)

    a=1

else:
    file_pattern_general = "C:\Yael\causal_inference\project\Project database\Patient_diagnosis_vitals_*.xlsx"
    file_list_general = FileLoader.load_files(file_pattern_general)

    file_pattern_admit = "C:\Yael\causal_inference\project\Admission\Patient_admission_*.xlsx"
    file_list_admit = FileLoader.load_files(file_pattern_admit)

    file_pattern_coag_rhythm = r"C:\Yael\causal_inference\project\blood_rhythm\Patient_blood_coagulation_rhythm_*.xlsx"
    file_list_coag_rhythm = FileLoader.load_files(file_pattern_coag_rhythm)

    file_pattern_diag_proc = r"C:\Yael\causal_inference\project\Patient_diagnoses_procedures\Patient_diagnoses_procedures_*.xlsx"
    file_list_diag_proc = FileLoader.load_files(file_pattern_diag_proc)



    df_list_general = []
    for file_diagnosis,file_admit, file_coag_rhythm, file_diag_proc in zip(file_list_general,file_list_admit,file_list_coag_rhythm,file_list_diag_proc):
        df_general = pd.read_excel(file_diagnosis, header=1)
        df_general = df_general.drop(['other_diagnoses','admission_type'],axis = 1)
        df_admit = pd.read_excel(file_admit, header=1)

        df_coag_rhythm = pd.read_excel(file_coag_rhythm, header=1)
        df_diag_proc = pd.read_excel(file_diag_proc, header=1, dtype={'other_diagnoses': str, 'procedures': str})

        result_df = pd.concat([df_general,df_admit,df_coag_rhythm,df_diag_proc], axis=1, ignore_index=False)
        new_df = pd.DataFrame()
        new_df = DataProcessor.process_test_data_columns(result_df,new_df, ['other_diagnoses','procedures','rhythm','hosp_blood', 'coagulation'])

        new_df['stay_duration'] = (result_df['discharge_time'] - result_df['admission_time']).dt.days.values
        new_df['discharge_location'] = result_df['discharge_location'].map(lambda x: 0 if 'home' in str(x).lower() else 1)
        new_df['patient_id'] = result_df['subject_id']

        new_df['age'] = result_df['calculated_age']
        new_df['race'] = result_df['subject_id']
        new_df['gender'] = result_df['gender'].map({'F': 0, 'M': 1})  # female = 0, male = 1
        new_df['married'] = result_df['marital_status'].map({'SINGLE': 0, 'DIVORCED': 0, 'MARRIED': 1, 'WIDOWED': 2,
                                                                 np.nan: 3})  # no = 0, yes = 1, widowed = 2, not known = 3
        new_df['months_to_death'] = result_df.apply(DataProcessor.calculate_months_passed, axis=1)
        new_df['death'] = pd.notna(result_df['dod']).astype(int)  # dead = 1, not dead = 0

        #new_df['weight'] = df_diagnosis['weight'].fillna(0)

        mapping_admission = mapping = {
    "EW EMER.": 4,               # Highest urgency
    "DIRECT EMER.": 4,
    "URGENT": 4,
    "OBSERVATION ADMIT": 3,
    "EU OBSERVATION": 3,
    "AMBULATORY OBSERVATION": 3,
    "DIRECT OBSERVATION": 3,
    "SURGICAL SAME DAY ADMISSION": 2,
    "ELECTIVE": 1                # Lowest urgency
}
        new_df['admission_type'] = result_df['admission_type'].map(mapping_admission) #.fillna(3).astype(int)


        # Mapping dictionary
        category_mapping = {
            'WHITE': 1,
            'WHITE - OTHER EUROPEAN': 1,
            'WHITE - EASTERN EUROPEAN': 1,
            'WHITE - RUSSIAN': 1,
            'BLACK/AFRICAN AMERICAN': 2,
            'BLACK/CAPE VERDEAN': 2,
            'BLACK/CARIBBEAN ISLAND': 2,
            'ASIAN': 3,
            'HISPANIC OR LATINO': 4,
            'PORTUGUESE': 5,
            'PATIENT DECLINED TO ANSWER': 6,
            'UNKNOWN': 6,
            'OTHER': 6
        }

        # Map the 'race' column and fill NaNs with 6
        new_df['race_mapped'] = result_df['race'].map(category_mapping).fillna(6)

        new_df = new_df.reset_index(drop=True)
        df_list_general.append(new_df)

    df_list_general = [df.reset_index(drop=True) for df in df_list_general]
    combined_df = pd.concat(df_list_general, ignore_index=True)

    combined_df = combined_df.reset_index(drop=True)


    combined_df = DataProcessor.remove_duplicate_columns(combined_df)


    #combined_df = combined_df.drop(columns=['other_diagnoses','procedures','admission_id','primary_diagnosis','hosp_blood','coagulation','rhythm','patient_weight'])
    combined_df.to_pickle(r'C:\Users\yaelp\OneDrive - Technion\causal_inference_project\Combined_data.pkl')



