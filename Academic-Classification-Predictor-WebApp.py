import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')

# Encoding map
encoding_maps = {
    'AgeAtEnrollment': {
        '18 - 20 years': 1,
        '21 - 23 years': 2,
        '24 - 26 years': 3,
        '26 years and above': 4
    },
    'Gender': ['Female', 'Male', 'Other'],
    'SyllabusMedium': ['Local Government Syllabus (Sri Lankan : English)',
                       'Local Government Syllabus (Sri Lankan : Tamil)',
                       'Local Government Syllabus (Sri Lankan : Sinhala)',
                       'Other',
                       'Cambridge International (CIE) or Edexcel (Pearson)'],
    'OLevelCoreModulePass': {
        "Yes, I achieved a minimum 'C' pass in all three subjects": 1,
        "No, I did not achieve a minimum 'C' pass in one or more of these subjects": 0
    },
    'ALStream': ['Mathematics Stream', 'Commerce Stream', 'Science Stream',
                 "Didn't take the A - Level examinations", 'Technology Stream', 'Arts Stream'],
    'ALevelCoreModulePass': {'Yes': 1, 'No': 0},
    'ALEnglishOrCourse': ['No, I did not follow A-Level English or any English course',
                          'Yes, I studied A-Level English',
                          'Yes, I completed an English course',
                          'Yes, I followed both A-Level English and completed an English course'],
    'PriorHigherEdu': {
        'No, I did not pursue any higher education prior to this degree': 0,
        'Foundation Program / Diploma related to Information Technology': 1
    },
    'GraduationYear': {
        2019: 1, 2020: 2, 2021: 3, 2022: 4, 2023: 5, 2024: 6
    },
    'SecondYearAvg': {
        'Below 40%': 1,
        '40% - 50%': 2,
        '51% - 60%': 3,
        '61% - 70%': 4,
        'Above 70%': 5
    },
    'InternshipCompleted': {
        'Yes, I completed the recommended internship': 1,
        'No, I did not complete the recommended internship': 0
    },
    'SatisfactionRating': [1, 2, 3, 4, 5],
    'FinalClassification': {
        'Pass Class': 1,
        'Second Class Lower': 2,
        'Second Class Upper': 3,
        'First Class': 4
    },
    'StressAnxietyLevel': [1, 2, 3, 4, 5],
    'PhysicalHealth': [1, 2, 3, 4, 5],
    'ChronicIllness': {'No': 0, 'Yes': 1},
    'ParentsEmployment': ['Both parents/guardians are employed',
                          'One parent/guardian is employed',
                          'Neither parents/guardian is employed'],
    'ParentsEducation': ["Higher education (Diploma or Bachelor's degree)",
                         "Postgraduate education (Master's degree or higher)",
                         'Completed A-Level',
                         'Completed O-Level',
                         'Prefer not to say'],
    'ParentsCohabitation': ['Both parents/guardians live together',
                            'Parents/guardians are separated but living independently',
                            'Both parents/guardians are deceased'],
    'HouseholdIncome': {
        'Below LKR 100,000': 1,
        'LKR 100,000 - 300,000': 2,
        'LKR 300,000 - 500,000': 3,
        'Above LKR 500,000': 4,
        'Prefer not to say': 0
    },
    'AccommodationType': ['Living with parents/guardians',
                          'Off-campus rented accommodation',
                          'Shared accommodation with friends or relatives'],
    'TransportMode': ['Driving / Driven by personal vehicle',
                      'Uber or other ride-hailing services',
                      'Public bus / train'],
    'TravelTime': {
        'Less than 30 minutes': 1,
        '30 minutes to 1 hour': 2,
        '1 hour to 1.5 hours': 3,
        '1.5 hours to 2 hours': 4,
        'More than 2 hours': 5
    },
    'EmployedDuringDegree': ['Yes, full-time employment',
                             'Yes, part-time employment',
                             'No, I was not employed'],
    'LeisureHoursPerWeek': {
        'Less than 5 hours': 1,
        '05 – 10 hours': 2,
        '10 – 20 hours': 3,
        'More than 20 hours': 4
    },
    'DailyScreenTime': {
        'Less than 2 hours': 1,
        '2 – 4 hours': 2,
        '5 – 7 hours': 3,
        '8 – 10 hours': 4,
        'More than 10 hours': 5
    }
}


def one_hot_encode(value, categories):
    return [1 if value == cat else 0 for cat in categories]


def multi_label_encode(selected, categories):
    return [1 if cat in selected else 0 for cat in categories]


def encode_input(user_input):
    encoded = []

    encoded.append(encoding_maps['AgeAtEnrollment'][user_input['AgeAtEnrollment']])
    encoded.extend(one_hot_encode(user_input['Gender'], encoding_maps['Gender']))
    encoded.extend(one_hot_encode(user_input['SyllabusMedium'], encoding_maps['SyllabusMedium']))
    encoded.append(encoding_maps['OLevelCoreModulePass'][user_input['OLevelCoreModulePass']])
    encoded.extend(one_hot_encode(user_input['ALStream'], encoding_maps['ALStream']))
    encoded.append(encoding_maps['ALevelCoreModulePass'][user_input['ALevelCoreModulePass']])
    encoded.extend(one_hot_encode(user_input['ALEnglishOrCourse'], encoding_maps['ALEnglishOrCourse']))
    encoded.append(encoding_maps['PriorHigherEdu'][user_input['PriorHigherEdu']])
    encoded.append(encoding_maps['GraduationYear'].get(user_input['GraduationYear'], 0))
    encoded.append(encoding_maps['SecondYearAvg'][user_input['SecondYearAvg']])
    encoded.append(encoding_maps['InternshipCompleted'][user_input['InternshipCompleted']])
    encoded.append(int(user_input['SatisfactionRating']))
    encoded.append(int(user_input['StressAnxietyLevel']))
    encoded.append(int(user_input['PhysicalHealth']))
    encoded.append(encoding_maps['ChronicIllness'][user_input['ChronicIllness']])
    encoded.extend(one_hot_encode(user_input['ParentsEmployment'], encoding_maps['ParentsEmployment']))
    encoded.extend(one_hot_encode(user_input['ParentsEducation'], encoding_maps['ParentsEducation']))
    encoded.extend(one_hot_encode(user_input['ParentsCohabitation'], encoding_maps['ParentsCohabitation']))
    encoded.append(encoding_maps['HouseholdIncome'][user_input['HouseholdIncome']])
    encoded.extend(one_hot_encode(user_input['AccommodationType'], encoding_maps['AccommodationType']))
    encoded.extend(multi_label_encode(user_input['TransportMode'], encoding_maps['TransportMode']))
    encoded.append(encoding_maps['TravelTime'][user_input['TravelTime']])
    encoded.extend(one_hot_encode(user_input['EmployedDuringDegree'], encoding_maps['EmployedDuringDegree']))
    encoded.append(encoding_maps['LeisureHoursPerWeek'][user_input['LeisureHoursPerWeek']])
    encoded.append(encoding_maps['DailyScreenTime'][user_input['DailyScreenTime']])

    return np.array(encoded).reshape(1, -1)


def main():
    st.title('Academic Degree Classification Predictor')

    # Replace your current select() function with this one
    def select(label, options):
        if isinstance(options, dict):
            options = list(options.keys())
        return st.selectbox(label, ['Select an option'] + options)

    def select_keyed(label, mapping):
        return st.selectbox(label, ['Select an option'] + list(mapping.keys()))

    AgeAtEnrollment = select_keyed(
        '01. What was your age at the time of enrollment in your current degree program? (Select the appropriate range)',
        encoding_maps['AgeAtEnrollment']
    )

    Gender = select(
        '02. What is your gender?',
        encoding_maps['Gender']
    )

    SyllabusMedium = select(
        '03. What type of syllabus and medium did you follow during your primary and secondary education? (Select the option that best applies to you)',
        encoding_maps['SyllabusMedium']
    )

    OLevelCoreModulePass = select_keyed(
        '04. Did you achieve a minimum \'C\' pass in Mathematics, English, and Computing at your Ordinary Level (O-Level) examinations?',
        encoding_maps['OLevelCoreModulePass']
    )

    ALStream = select(
        '06. Which stream did you study for your Advanced Level (A-Level) examination? (Select the option that best applies to you.)',
        encoding_maps['ALStream']
    )

    ALevelCoreModulePass = select_keyed(
        '07. Did you achieve a minimum of three \'C\' passes in your Advanced Level (A-Level) examination? (Please select the option that best matches your overall results.)',
        encoding_maps['ALevelCoreModulePass']
    )

    ALEnglishOrCourse = select(
        '09. Did you follow A-Level English or complete an English course prior to starting your degree program?',
        encoding_maps['ALEnglishOrCourse']
    )

    PriorHigherEdu = select_keyed(
        '10. Did you pursue any higher education program prior to starting your current degree?',
        encoding_maps['PriorHigherEdu']
    )

    GraduationYear = select_keyed(
        '11. In which year did you graduate from the Informatics Institute of Technology (IIT) Sri Lanka?',
        encoding_maps['GraduationYear']
    )

    SecondYearAvg = select_keyed(
        '12. What was your average percentage for your second year of studies? (Select the range that best reflects your average score)',
        encoding_maps['SecondYearAvg']
    )

    InternshipCompleted = select_keyed(
        '13. Did you complete the recommended degree-related internship during your third year of studies that aligns with the career opportunities provided by your degree program?',
        encoding_maps['InternshipCompleted']
    )

    SatisfactionRating = select(
        '14. After returning for your final year of studies, how would you rate your satisfaction with your degree program and your overall career path? (Use the scale below to rate your satisfaction.)',
        list(map(str, encoding_maps['SatisfactionRating']))
    )


    StressAnxietyLevel = select(
        '15. How would you rate your overall stress and anxiety levels during your time in the degree program? (Use the scale below to rate your experience.)',
        list(map(str, encoding_maps['StressAnxietyLevel']))
    )

    PhysicalHealth = select(
        '16. How would you rate your overall physical health during your time in the degree program? (Use the scale below to rate your experience.)',
        list(map(str, encoding_maps['PhysicalHealth']))
    )

    ChronicIllness = select_keyed(
        '17. Did you have any chronic illnesses that hindered your academic performance during your degree program? (For example, asthma, diabetes, etc.)',
        encoding_maps['ChronicIllness']
    )

    ParentsEmployment = select(
        '18. What is the employment status of your parents or guardians?',
        encoding_maps['ParentsEmployment']
    )

    ParentsEducation = select(
        '19. What is the highest level of education attained by at least one of your parents or guardians? (Select the option that best applies to both parents or guardians)',
        encoding_maps['ParentsEducation']
    )

    ParentsCohabitation = select(
        '20. What is the marital status or cohabitation status of your parents or guardians?',
        encoding_maps['ParentsCohabitation']
    )

    HouseholdIncome = select_keyed(
        '21. What was the approximate monthly household income during your time as a student? (Select the option that best applies to your household)',
        encoding_maps['HouseholdIncome']
    )

    AccommodationType = select(
        '22. What type of accommodation did you use during your degree program?',
        encoding_maps['AccommodationType']
    )

    TransportMode = st.multiselect(
        '23. What mode of transport did you use to travel from your accommodation to the university campus?',
        encoding_maps['TransportMode']
    )

    TravelTime = select_keyed(
        '24. How long did it typically take to travel from your accommodation to the university campus?',
        encoding_maps['TravelTime']
    )

    EmployedDuringDegree = select(
        '25. Were you employed while pursuing your degree?',
        encoding_maps['EmployedDuringDegree']
    )

    LeisureHoursPerWeek = select_keyed(
        '26. How many hours did you spend on leisure activities per week during your degree program?',
        encoding_maps['LeisureHoursPerWeek']
    )

    DailyScreenTime = select_keyed(
        '27. How many hours did you spend on screen time per day during your degree program? (Including television, computer, and mobile devices)',
        encoding_maps['DailyScreenTime']
    )

    if st.button('Predict Degree Classification'):
        # Check for missing inputs
        inputs = {
            'AgeAtEnrollment': AgeAtEnrollment,
            'Gender': Gender,
            'SyllabusMedium': SyllabusMedium,
            'OLevelCoreModulePass': OLevelCoreModulePass,
            'ALStream': ALStream,
            'ALevelCoreModulePass': ALevelCoreModulePass,
            'ALEnglishOrCourse': ALEnglishOrCourse,
            'PriorHigherEdu': PriorHigherEdu,
            'GraduationYear': GraduationYear,
            'SecondYearAvg': SecondYearAvg,
            'InternshipCompleted': InternshipCompleted,
            'SatisfactionRating': SatisfactionRating,
            'StressAnxietyLevel': StressAnxietyLevel,
            'PhysicalHealth': PhysicalHealth,
            'ChronicIllness': ChronicIllness,
            'ParentsEmployment': ParentsEmployment,
            'ParentsEducation': ParentsEducation,
            'ParentsCohabitation': ParentsCohabitation,
            'HouseholdIncome': HouseholdIncome,
            'AccommodationType': AccommodationType,
            'TransportMode': TransportMode,
            'TravelTime': TravelTime,
            'EmployedDuringDegree': EmployedDuringDegree,
            'LeisureHoursPerWeek': LeisureHoursPerWeek,
            'DailyScreenTime': DailyScreenTime
        }

        if 'Select an option' in inputs.values() or TransportMode == []:
            st.error('Please fill all fields before prediction.')
            return

        # Encode inputs
        try:
            encoded_input = encode_input(inputs)
            scaled_input = scaler.transform(encoded_input)
            prediction = model.predict(scaled_input)

            inverse_final_classification = {v: k for k, v in encoding_maps['FinalClassification'].items()}
            predicted_label = inverse_final_classification.get(prediction[0], 'Unknown')

            st.success(f'Predicted degree classification: {predicted_label}')
        except Exception as e:
            st.error(f'Error during prediction: {str(e)}')


if __name__ == '__main__':
    main()

