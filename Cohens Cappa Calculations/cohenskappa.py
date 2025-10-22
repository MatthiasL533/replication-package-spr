import pandas as pd
import numpy as np
import sys
from itertools import combinations
from sklearn.metrics import cohen_kappa_score
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def parse_violations(value):
    """Convert violation string to set of violations."""
    if pd.isna(value) or value == 'No violation':
        return set()
    return set(v.strip() for v in str(value).split(','))


def fleiss_kappa(ratings_matrix):
    """
    Calculate Fleiss' Kappa for multiple raters.
    
    Args:
        ratings_matrix: 2D array where rows are subjects and columns are categories
                       Each cell contains the count of raters who assigned that category
    """
    n_subjects, n_categories = ratings_matrix.shape
    n_raters = ratings_matrix.sum(axis=1)[0]  # Total raters per subject
    
    # Calculate p_j (proportion of all assignments to category j)
    p_j = ratings_matrix.sum(axis=0) / (n_subjects * n_raters)
    
    # Calculate P_i (extent of agreement for subject i)
    P_i = (np.sum(ratings_matrix ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    
    # Calculate P_bar (mean of P_i)
    P_bar = P_i.mean()
    
    # Calculate P_e_bar (expected agreement by chance)
    P_e_bar = np.sum(p_j ** 2)
    
    # Calculate Fleiss' Kappa
    if P_e_bar == 1:
        return 0.0
    kappa = (P_bar - P_e_bar) / (1 - P_e_bar)
    
    return kappa


def prepare_data_for_kappa(df, col1, col2):
    """Prepare two columns for Cohen's Kappa calculation."""
    violations1 = df[col1].apply(parse_violations)
    violations2 = df[col2].apply(parse_violations)
    
    # Get all unique violation types
    all_violations = set()
    for v1, v2 in zip(violations1, violations2):
        all_violations.update(v1)
        all_violations.update(v2)
    
    # Create binary vectors for each violation type
    labels1 = []
    labels2 = []
    
    for v1, v2 in zip(violations1, violations2):
        # Create a combined label representing the set of violations
        label1 = ','.join(sorted(v1)) if v1 else 'No violation'
        label2 = ','.join(sorted(v2)) if v2 else 'No violation'
        labels1.append(label1)
        labels2.append(label2)
    
    return labels1, labels2


def calculate_all_cohens_kappa(df, evaluator_columns):
    """Calculate Cohen's Kappa for all pairs of evaluators."""
    results = []
    
    for col1, col2 in combinations(evaluator_columns, 2):
        labels1, labels2 = prepare_data_for_kappa(df, col1, col2)
        kappa = cohen_kappa_score(labels1, labels2)
        results.append({
            'Rater 1': col1,
            'Rater 2': col2,
            'Cohen\'s Kappa': round(kappa, 4)
        })
    
    return pd.DataFrame(results)


def calculate_fleiss_kappa_for_dataset(df, evaluator_columns):
    """Calculate Fleiss' Kappa for all evaluators."""
    # Get all unique violation combinations across all evaluators
    all_violation_combos = set()
    for col in evaluator_columns:
        violations = df[col].apply(lambda x: ','.join(sorted(parse_violations(x))) if parse_violations(x) else 'No violation')
        all_violation_combos.update(violations.unique())
    
    violation_list = sorted(all_violation_combos)
    violation_to_idx = {v: i for i, v in enumerate(violation_list)}
    
    # Create ratings matrix
    n_subjects = len(df)
    n_categories = len(violation_list)
    ratings_matrix = np.zeros((n_subjects, n_categories))
    
    for idx, row in df.iterrows():
        for col in evaluator_columns:
            violations = parse_violations(row[col])
            violation_combo = ','.join(sorted(violations)) if violations else 'No violation'
            category_idx = violation_to_idx[violation_combo]
            ratings_matrix[idx, category_idx] += 1
    
    kappa = fleiss_kappa(ratings_matrix)
    return kappa, violation_list


def export_to_excel(cohens_results, fleiss_kappa_value, evaluator_columns, df, filename, output_filename):
    """Export results to CSV files that can be opened in Excel."""
    
    # Create summary statistics DataFrame
    summary_stats = pd.DataFrame({
        'Metric': [
            'Dataset',
            'Number of Raters',
            'Number of User Stories',
            'Fleiss\' Kappa',
            '',
            'Mean Cohen\'s Kappa',
            'Median Cohen\'s Kappa',
            'Std Dev Cohen\'s Kappa',
            'Min Cohen\'s Kappa',
            'Max Cohen\'s Kappa'
        ],
        'Value': [
            filename,
            len(evaluator_columns),
            len(df),
            round(fleiss_kappa_value, 4),
            '',
            round(cohens_results['Cohen\'s Kappa'].mean(), 4),
            round(cohens_results['Cohen\'s Kappa'].median(), 4),
            round(cohens_results['Cohen\'s Kappa'].std(), 4),
            round(cohens_results['Cohen\'s Kappa'].min(), 4),
            round(cohens_results['Cohen\'s Kappa'].max(), 4)
        ]
    })
    
    # Export Cohen's Kappa results
    cohens_output = f"{output_filename}_cohens_kappa.csv"
    cohens_results.to_csv(cohens_output, index=False)
    print(f"✓ Cohen's Kappa results exported to: {cohens_output}")
    
    # Export summary statistics
    summary_output = f"{output_filename}_summary.csv"
    summary_stats.to_csv(summary_output, index=False)
    print(f"✓ Summary statistics exported to: {summary_output}")
    
    # Create interpretation guide
    interpretation = pd.DataFrame({
        'Kappa Range': ['< 0.00', '0.00 - 0.20', '0.21 - 0.40', '0.41 - 0.60', '0.61 - 0.80', '0.81 - 1.00'],
        'Interpretation': ['Poor agreement', 'Slight agreement', 'Fair agreement', 'Moderate agreement', 'Substantial agreement', 'Almost perfect agreement']
    })
    interpretation_output = f"{output_filename}_interpretation.csv"
    interpretation.to_csv(interpretation_output, index=False)
    print(f"✓ Interpretation guide exported to: {interpretation_output}")
    
    # Create best and worst pairs
    best_worst = pd.DataFrame()
    best_worst['Category'] = ['Best Agreement'] * 3 + [''] + ['Worst Agreement'] * 3
    
    best = cohens_results.nlargest(3, 'Cohen\'s Kappa').reset_index(drop=True)
    worst = cohens_results.nsmallest(3, 'Cohen\'s Kappa').reset_index(drop=True)
    
    combined = pd.concat([best, pd.DataFrame([['', '', '']], columns=best.columns), worst], ignore_index=True)
    combined.insert(0, 'Category', best_worst['Category'])
    
    best_worst_output = f"{output_filename}_best_worst.csv"
    combined.to_csv(best_worst_output, index=False)
    print(f"✓ Best/worst agreements exported to: {best_worst_output}")
    
    print(f"\n✓ All results exported! You can now open these CSV files in Excel.")


def main():
    # Get filename from command line argument or use default
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = 'total_dataset.csv'
        print(f"No filename provided. Using default: {filename}")
        print(f"Usage: python {sys.argv[0]} <filename.csv>")
        print()

    # Read the CSV file with encoding handling
    try:
        df = pd.read_csv(filename, sep=';', encoding='utf-8')
    except UnicodeDecodeError:
        # Try alternative encodings
        try:
            df = pd.read_csv(filename, sep=';', encoding='latin-1')
        except:
            df = pd.read_csv(filename, sep=';', encoding='cp1252')
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)
    
    # Define evaluator columns
    evaluator_columns = [
        'Our Evaluation',
        'AQUSA Evaluation',
        'OpenAIs GPT-5 Evaluation',
        'Claude 4.5 Sonnet Evaluation',
        'Gemini 2.5 Flash Evaluation',
        'DeepSeek-V3.1 Evaluation'
    ]
    
    print("=" * 80)
    print(f"Dataset: {filename}")
    print("USER STORY QUALITY EVALUATION - INTER-RATER AGREEMENT ANALYSIS")
    print("=" * 80)
    print()
    
    # Calculate Cohen's Kappa for all pairs
    print("COHEN'S KAPPA (Pairwise Agreement)")
    print("-" * 80)
    cohens_results = calculate_all_cohens_kappa(df, evaluator_columns)
    print(cohens_results.to_string(index=False))
    print()
    
    # Interpretation guide
    print("\nInterpretation Guide for Kappa Values:")
    print("  < 0.00: Poor agreement")
    print("  0.00 - 0.20: Slight agreement")
    print("  0.21 - 0.40: Fair agreement")
    print("  0.41 - 0.60: Moderate agreement")
    print("  0.61 - 0.80: Substantial agreement")
    print("  0.81 - 1.00: Almost perfect agreement")
    print()
    
    # Calculate Fleiss' Kappa
    print("=" * 80)
    print("FLEISS' KAPPA (Overall Agreement Across All Raters)")
    print("-" * 80)
    fleiss_kappa_value, categories = calculate_fleiss_kappa_for_dataset(df, evaluator_columns)
    print(f"Fleiss' Kappa: {fleiss_kappa_value:.4f}")
    print()
    print(f"Number of raters: {len(evaluator_columns)}")
    print(f"Number of subjects (user stories): {len(df)}")
    print(f"Number of unique evaluation categories: {len(categories)}")
    print()
    
    # Summary statistics
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("-" * 80)
    print(f"Mean Cohen's Kappa: {cohens_results['Cohen\'s Kappa'].mean():.4f}")
    print(f"Median Cohen's Kappa: {cohens_results['Cohen\'s Kappa'].median():.4f}")
    print(f"Std Dev Cohen's Kappa: {cohens_results['Cohen\'s Kappa'].std():.4f}")
    print(f"Min Cohen's Kappa: {cohens_results['Cohen\'s Kappa'].min():.4f}")
    print(f"Max Cohen's Kappa: {cohens_results['Cohen\'s Kappa'].max():.4f}")
    print()
    
    # Best and worst agreement pairs
    print("Best Agreement (Highest Cohen's Kappa):")
    best = cohens_results.nlargest(3, 'Cohen\'s Kappa')
    print(best.to_string(index=False))
    print()
    
    print("Weakest Agreement (Lowest Cohen's Kappa):")
    worst = cohens_results.nsmallest(3, 'Cohen\'s Kappa')
    print(worst.to_string(index=False))
    print()
    
    print("=" * 80)
    print()
    
    # Export to CSV files for Excel
    print("EXPORTING RESULTS TO CSV FILES FOR EXCEL")
    print("-" * 80)
    
    # Create output filename based on input filename
    base_name = filename.rsplit('.', 1)[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{base_name}_kappa_results_{timestamp}"
    
    export_to_excel(cohens_results, fleiss_kappa_value, evaluator_columns, df, filename, output_filename)
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()