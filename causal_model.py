# Core Libraries
import os
import random
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

# Scikit-Learn Models and Tools
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LogisticRegression, LinearRegression, LassoCV, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.pipeline import make_pipeline

# Causal Inference Models
from econml.metalearners import XLearner
from econml.dml import LinearDML
from econml.dr import DRLearner
from causalml.inference.tree import CausalTreeRegressor

# Seed setting for reproducibility
np.random.seed(0)
random.seed(0)


#utility functions
def create_output_folder(folder_name="causal_analysis_plots"):
    """Create a folder to store outputs if it doesn't exist."""
    base_path = r'C:\Users\yaelp\OneDrive - Technion\causal_inference_project'
    path = os.path.join(base_path, folder_name)
    os.makedirs(path, exist_ok=True)
    return path

def bootstrap_ci(func, data_indices, n_bootstrap=1000, alpha=0.05):
    """Calculate bootstrap confidence intervals using resampling."""
    estimates = []

    # Perform bootstrap resampling
    for _ in range(n_bootstrap):
        resampled_indices = np.random.choice(data_indices, size=len(data_indices), replace=True)
        estimate = func(resampled_indices)
        estimates.append(estimate)

    # Compute confidence intervals
    lower_bound = np.percentile(estimates, 100 * alpha / 2)
    upper_bound = np.percentile(estimates, 100 * (1 - alpha / 2))
    return np.mean(estimates), (lower_bound, upper_bound)



#causal analysis functions
def cate_by_feature_groups(X, T, Y, feature_name,feature_ind, bins, model, model_name, feature_group_labels,output_name):
    # Fit the model based on its type (causal models like X-Learner or causal trees)
    if hasattr(model, 'fit') and 'T' in model.fit.__code__.co_varnames:  # Causal models like X-Learner
        model.fit(Y=Y, T=T, X=X)
    elif isinstance(model, CausalTreeRegressor):  # CausalTreeRegressor requires the treatment argument
        model.fit(X=X, treatment=T, y=Y)
    else:  # Standard scikit-learn models
        model.fit(X, Y)

    # Calculate CATE or treatment effects based on the model type
    if hasattr(model, 'effect'):  # Causal models like X-Learner
        cate = model.effect(X)
    elif isinstance(model, CausalTreeRegressor):  # CausalTreeRegressor uses predict() to estimate treatment effects
        cate = model.predict(X)
    else:
        cate = model.predict(X)

    feature_values = X[feature_name]
    group_labels = list(feature_group_labels.values())[int(feature_ind)]
    group_name = list(feature_group_labels.keys())[int(feature_ind)]
    # Retrieve the appropriate labels for the current feature from feature_group_labels


    # Use pd.cut for continuous features with bins, otherwise pd.qcut for categorical
    if isinstance(bins, list):
        try:
            # Extend the last bin to include any values beyond the highest bin value
            max_bin = bins[-1]  # Get the highest bin value
            extended_bins = bins[:-1] + [np.inf]  # Extend the last bin to infinity

            # Group values using the extended bins
            groups = pd.cut(feature_values, bins=extended_bins, labels=[i for i in range(1, len(bins))],
                            include_lowest=True)
        except ValueError as e:
            print(f"Error during pd.cut: {e}")
            groups = pd.Series([None] * len(feature_values))  # Assign None if an error occurs
    else:
        try:
            # Group into quantiles, handle insufficient unique values
            groups = pd.qcut(feature_values.rank(method="first"), bins, labels=[i for i in range(1, bins + 1)])
        except ValueError as e:
            print(f"Error during pd.qcut: {e}")
            groups = pd.Series([None] * len(feature_values))  # Assign None if an error occurs

    # Get full CATE vector for each group
    cate_by_group = X.groupby(groups).apply(lambda group: cate[group.index])
    # Initialize lists for storing CIs
    ci_lowers = []
    ci_uppers = []

    # Calculate bootstrap CI for each group
    for group_idx in cate_by_group.index:
        group_indices = groups[groups == group_idx].index
        group_cate = cate[group_indices]

        # Bootstrap CI for this group's CATE
        mean_estimate, (ci_lower, ci_upper) = bootstrap_ci(
            lambda X_group: np.mean(X_group), group_cate
        )
        ci_lowers.append(ci_lower)
        ci_uppers.append(ci_upper)


    if group_labels and len(group_labels) == len(cate_by_group.index):
        # Use group labels if available and the lengths match
        labels_for_plot = [list(group_labels.values())[i] for i in range(len(cate_by_group))]
    else:
        # Use the bin ranges as labels if no group labels are provided
        if isinstance(bins, list):  # This assumes bins is a list of bin edges
            labels_for_plot = [f"{bins[i]} - {bins[i + 1]}" for i in range(len(bins) - 1)]
        else:
            # If bins is not a list, it is categorical data, just show the group index
            labels_for_plot = list(cate_by_group.index)

    cate_values = list(cate_by_group.values)
    group_indices = np.array(cate_by_group.index)
    cate_means = np.array([np.mean(arr) for arr in cate_values])
    yerr = np.vstack([cate_means - ci_lowers, ci_uppers - cate_means])

    #Step 4: Plot CATE with CI Error Bars
    plt.figure(figsize=(10, 6))
    bars = plt.bar(group_indices, cate_means, color='skyblue', alpha=0.7, edgecolor='black', capsize=5)

    # Add error bars to each bar
    plt.errorbar(
        group_indices, cate_means, yerr=yerr,
        fmt='none', ecolor='black', capsize=5, elinewidth=2
    )

    plt.xlabel(f'{group_name} Groups')
    plt.ylabel('Average CATE with 95% CI')
    plt.xticks(ticks=group_indices, labels=labels_for_plot, rotation=45)
    plt.title(f'{model_name}: CATE by {feature_name} with Confidence Intervals')
    plt.tight_layout()

    # Save the plot
    folder = create_output_folder(f"{model_name}_{output_name}_plots")
    plt.savefig(f"{folder}/{model_name}_cate_by_{feature_name}.png")
    plt.close()

    return cate_by_group

# Calculate and plot CATE for all features
def cate_for_all_features(X, T, Y, features, model, model_name, bins_dict, feature_group_labels, output_name):
    results = {}
    for feature_ind, feature in enumerate(features):
        bins = bins_dict.get(feature, 5)  # Default bins
        cate_by_feature = cate_by_feature_groups(X, T, Y, feature, feature_ind, bins, model, model_name, feature_group_labels, output_name)
        results[feature] = cate_by_feature
    return results


#statistical tests and saving results
def check_normality_and_perform_tests(cate_by_features):
    test_results = []
    for feature, cate in cate_by_features.items():
        normal = all(stats.shapiro(vec)[1] > 0.05 for vec in cate)
        stat, p_val = (stats.f_oneway if normal else stats.kruskal)(*cate)
        test_results.append({"Feature": feature, "Test": "ANOVA" if normal else "Kruskal-Wallis", "p-value": p_val})
    return pd.DataFrame(test_results)



def save_results_to_excel(model_name, ate, att, cate_results):
    folder = create_output_folder(f"{model_name}_outputs")
    path = os.path.join(folder, f"{model_name}_results.xlsx")
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"ATE": [ate], "ATT": [att]}).to_excel(writer, sheet_name='ATE_ATT')
        for feature, cate in cate_results.items():
            pd.DataFrame(cate).to_excel(writer, sheet_name=f'CATE_{feature}')



#model specific functions
def x_learner_analysis(X, T, Y, features, bins_dict, feature_group_labels, output_name):
    # Define the base learner (Ridge regression model)
    base_learner = make_pipeline(StandardScaler(), Ridge(alpha=1.0))

    # Handle binary treatment assignment with a logistic regression model
    treatment_model = LogisticRegression(
        solver='saga',      # Robust solver
        max_iter=5000,      # Increased iterations for convergence
        random_state=0,     # Set random seed for reproducibility
        C=0.1               # Regularization to avoid overfitting
    )

    # Initialize the X-Learner with the models for Y and T
    x_learner = XLearner(
        models=base_learner,
        propensity_model=treatment_model
    )

    # Run model analysis with tests and return results
    return model_analysis_with_tests(
        x_learner, X, T, Y, features, bins_dict,
        "x_learner", feature_group_labels, output_name
    )


# Causal Tree Analysis
def causal_tree_analysis(X, T, Y, features, bins_dict, feature_group_labels, compare, output_name):
    # Initialize the Causal Tree Regressor
    causal_tree_model = CausalTreeRegressor(max_depth=20, random_state=0)

    # Perform model analysis based on the comparison type
    if compare == 'models':
        return model_analysis_with_tests(
            causal_tree_model, X, T, Y, features, bins_dict,
            "causal_tree", feature_group_labels, output_name
        )
    else:
        return model_analysis_with_tests(
            causal_tree_model, X, T, Y, features, bins_dict,
            "causal_tree_different_outcomes", feature_group_labels, output_name
        )


# Doubly Robust Analysis
def doubly_robust_analysis(X, T, Y, features, bins_dict, feature_group_labels, compare, output_name):
    # Configure RandomForestRegressor with tighter constraints to avoid overfitting
    rf_model = RandomForestRegressor(
        n_estimators=100,          # Number of trees in the forest
        max_depth=5,               # Limit depth to control complexity
        min_samples_split=15,      # Minimum samples to split an internal node
        min_samples_leaf=10,       # Minimum samples at a leaf node
        random_state=0             # Set random seed for reproducibility
    )

    # Logistic Regression for treatment assignment with L2 regularization
    treatment_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            solver='lbfgs',          # Standard solver for logistic regression
            max_iter=10000,          # Increased iterations to ensure convergence
            C=1.0,                   # L2 regularization strength
            random_state=0           # Set random seed for reproducibility
        )
    )

    # Initialize the LinearDML model with appropriate parameters
    dr_learner = LinearDML(
        model_y=rf_model,                  # Model for outcome prediction
        model_t=treatment_model,           # Model for treatment assignment
        featurizer=PolynomialFeatures(degree=1, include_bias=False),
        discrete_treatment=True,           # Specify that treatment is discrete
        random_state=0                     # Set random seed for reproducibility
    )

    # Perform model analysis based on the comparison type
    if compare == 'models':
        return model_analysis_with_tests(
            dr_learner, X, T, Y, features, bins_dict,
            "doubly_robust", feature_group_labels, output_name
        )
    else:
        return model_analysis_with_tests(
            dr_learner, X, T, Y, features, bins_dict,
            "doubly_different_outcomes", feature_group_labels, output_name
        )


    if compare == 'models':
        return model_analysis_with_tests(dr_learner, X, T, Y, features, bins_dict, "doubly_robust", feature_group_labels,output_name)
    else:
        return model_analysis_with_tests(dr_learner, X, T, Y, features, bins_dict, "doubly_different_outcomes",
                                         feature_group_labels,output_name)


def plot_ate_att_comparison(results_dict,output_name = '', comparison_type='models'):
    """
    Plots a comparison of ATE and ATT for different models or outcomes.

    Parameters:
    - results_dict: Dictionary where keys are the labels for different models/outcomes,
      and values are tuples of (ate_value, att_value).
    - comparison_type: A string that specifies whether the comparison is between 'models' or 'outcomes'.
                       This adjusts the plot title and folder name accordingly.
    """
    # Extract labels and their corresponding ATE and ATT values
    labels = list(results_dict.keys())
    ate_values = [results_dict[label][0] for label in labels]  # ATE values
    ate_cis = [results_dict[label][1] for label in labels]     # ATE CIs
    att_values = [results_dict[label][2] for label in labels]  # ATT values
    att_cis = [results_dict[label][3] for label in labels]     # ATT CIs

    # Calculate error bars (yerr) for ATE and ATT
    ate_errors = np.array([
        [ate - ci[0], ci[1] - ate] for ate, ci in zip(ate_values, ate_cis)
    ]).T  # Transpose to match yerr shape

    att_errors = np.array([
        [att - ci[0], ci[1] - att] for att, ci in zip(att_values, att_cis)
    ]).T  # Transpose to match yerr shape


    # Create a plot comparing ATE and ATT for the given models/outcomes
    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    index = np.arange(len(labels))

    # Plot ATE with error bars
    plt.bar(index, ate_values, width=bar_width, label='ATE', alpha=0.7, color='b', yerr=ate_errors, capsize=5)

    # Plot ATT with error bars next to ATE
    plt.bar(index + bar_width, att_values, width=bar_width, label='ATT', alpha=0.7, color='r', yerr=att_errors, capsize=5)

    # Set title, labels, and legend based on the comparison type
    title = f"Comparison of ATE and ATT for Different {comparison_type.capitalize()}"
    plt.xlabel(comparison_type.capitalize())
    plt.ylabel('Effect Value')
    plt.title(title)
    plt.xticks([i + bar_width / 2 for i in index], labels, rotation=45, ha='right')
    plt.legend()

    # Adjust folder name based on the comparison type
    folder_name = f"{comparison_type}_{output_name}_comparison_plots"
    folder = create_output_folder(folder_name)
    output_path = os.path.join(folder, f'ate_att_comparison_plot_{comparison_type}.png')

    # Save the plot to a file
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"ATE vs ATT comparison plot saved to {output_path}")



# Run models and calculate ATE, ATT, and CATE
def model_analysis_with_tests(model, X, T, Y, features, bins_dict, model_name, feature_group_labels, output_name):
    # Fit the model and calculate ATE and ATT
    if hasattr(model, 'fit') and 'T' in model.fit.__code__.co_varnames:
        model.fit(Y=Y, T=T, X=X)
        ate = model.ate(X)
        att = model.ate(X[T == 1])
        cate = model.effect(X)

        # Calculate Bootstrap Confidence Intervals
        ate_mean, ate_ci = bootstrap_ci(
            lambda idx: np.mean(model.effect(X.iloc[idx])),  # Use resampled X to compute ATE
            np.arange(len(X)),  # Pass indices to bootstrap_ci for resampling
            n_bootstrap=1000
        )

        att_mean, att_ci = bootstrap_ci(
            lambda idx: np.mean(model.effect(X.iloc[idx][T[idx] == 1])),
            np.arange(len(X)),
            n_bootstrap=1000
        )



    elif isinstance(model, CausalTreeRegressor):
        model.fit(X=X, treatment=T, y=Y)
        cate = model.predict(X)
        ate = np.mean(cate)
        att = np.mean(cate[T == 1])

        # Calculate Bootstrap Confidence Intervals
        ate_mean, ate_ci = bootstrap_ci(
            lambda idx: np.mean(model.predict(X.iloc[idx])),  # Use resampled X to compute ATE
            np.arange(len(X)),  # Pass indices to bootstrap_ci for resampling
            n_bootstrap=1000
        )

        att_mean, att_ci = bootstrap_ci(
            lambda idx: np.mean(model.predict(X.iloc[idx][T[idx] == 1])),
            np.arange(len(X)),
            n_bootstrap=1000
        )

    else:
        model_treated = model.fit(X[T == 1], Y[T == 1])
        model_control = model.fit(X[T == 0], Y[T == 0])
        Y_treated_pred = model_treated.predict(X)
        Y_control_pred = model_control.predict(X)
        cate = Y_treated_pred - Y_control_pred
        ate = np.mean(cate)
        att = np.mean(cate[T == 1])

        # Calculate Bootstrap Confidence Intervals
        ate_mean, ate_ci = bootstrap_ci(
            lambda idx: np.mean(model_treated.predict(X.iloc[idx])-model_control.predict(X.iloc[idx])),  # Use resampled X to compute ATE
            np.arange(len(X)),  # Pass indices to bootstrap_ci for resampling
            n_bootstrap=1000
        )

        att_mean, att_ci = bootstrap_ci(
            lambda idx: np.mean(model_treated.predict(X.iloc[idx][T[idx] == 1])-model_control.predict(X.iloc[idx][T[idx] == 1])),
            np.arange(len(X)),
            n_bootstrap=1000
        )




    print(f"ATE: {ate_mean} (95% CI: {ate_ci})")
    print(f"ATT: {att_mean} (95% CI: {att_ci})")

    # Generate CATE by feature
    cate_by_all_features = cate_for_all_features(X, T, Y, features, model, model_name, bins_dict, feature_group_labels, output_name)

    # Perform statistical tests on the CATE results
    test_results = check_normality_and_perform_tests(cate_by_all_features)

    # Save results to Excel
    save_results_to_excel(f"{model_name}_results", ate_mean, att_mean, cate_by_all_features, model_name, output_name)


    return ate_mean,ate_ci, att_mean,att_ci, cate_by_all_features



