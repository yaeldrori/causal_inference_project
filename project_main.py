import pickle
import os
from causal_model import x_learner_analysis, causal_tree_analysis, doubly_robust_analysis,plot_ate_att_comparison


if __name__ == "__main__":

    #output_path = r'C:\Yael\causal_inference\project\Combined_data.pkl'
    output_path =r'C:\Users\yaelp\OneDrive - Technion\causal_inference_project\Combined_data.pkl'

    if os.path.exists(output_path):
        with open(output_path, 'rb') as file:
            data = pickle.load(file)

    T = data['cardioversion_procedure'].values
    output_name = 'death' #'stay_duration' # 'death' #'rhythm_discharge' # 'coag_discharge' #'death', 'rhythm_discharge','stay_duration'
    Y = data[output_name].values

    features = ['age','gender','married','cardiac_diagnosis','cardiac_procedure',
                'admission_type','rhythm_admit','coag_admit','blood_admit',
               ]
    X = data[features]
    columns_with_nan = X.columns[X.isna().any()].tolist()


    bins_dict = {
        'age': [20, 30, 40, 50, 60, 70],  # Custom bins for age
        'gender': 2,  # 2 categories for gender
        'married': 4,  # 4 categories for marital status (0-3)
        'cardiac_diagnosis': 2,  # Binary diagnosis (0 or 1)
        'cardiac_procedure': 2,  # Binary procedure (0 or 1)
        'admission_type': 4,  # 4 categories (1-4)
        'rhythm_admit': 3,  # 3 categories for rhythm (0-2)
        'coag_admit': 3,  # Binary for coagulation state
        'blood_admit': 3  # Binary for blood state
    }

    feature_group_labels = {
        'age': None,  # Used as continuous value, no categorical labels needed
        'gender': {0: 'Female', 1: 'Male'},
        'married': {0: 'Single/Divorced', 1: 'Married', 2: 'Widowed', 3: 'Not known'},
        'cardiac diagnosis': {0: 'No cardiac-related diagnosis', 1: 'Cardiac-related diagnosis'},
        'cardiac procedure': {0: 'No cardiac-related procedure', 1: 'Cardiac-related procedure'},
        'admission type': {1: 'Elective',2: 'Surgical Same-Day Admission',3:'Observation', 4: 'Urgent'},
        'rhythm at admission': {-1: 'Sinus Rhythm', 0: 'No Data', 1: 'Abnormal Rhythms'},
        'coagulation values at admission': {-1: 'Normal', 0: 'No Data', 1: 'Abnormal'},
        'blood tests at admission': {-1: 'Normal', 0: 'No Data', 1: 'Abnormal'}

    }


    # Run X-Learner Analysis
    ate_x,ate_ci_x, att_x,att_ci_x, cate_x = x_learner_analysis(X, T, Y, features, bins_dict, feature_group_labels,output_name)

    # Run Causal Tree Analysis
    ate_tree, ate_ci_tree, att_tree,att_ci_tree, cate_tree = causal_tree_analysis(X, T, Y, features, bins_dict, feature_group_labels,'models',output_name)

    # # Run Doubly Robust Analysis
    ate_dr, ate_ci_dr, att_dr, att_ci_dr, cate_dr = doubly_robust_analysis(X, T, Y, features, bins_dict, feature_group_labels,'models',output_name)
    #
    results_dict = {
         'X-Learner': (ate_x,ate_ci_x, att_x,att_ci_x),
         'Causal Tree': (ate_tree,ate_ci_tree, att_tree,att_ci_tree),
         'Doubly Robust': (ate_dr,ate_ci_dr, att_dr,att_ci_dr),
     }

    plot_ate_att_comparison(results_dict, output_name = output_name)


