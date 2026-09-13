"""
Test script for the Data Quality Scoring and Reporting Engine.

1. Loads the clean dataset (CSV).
2. Loads the dirty dataset (Excel).
3. Executes all 10 quality checks from src/data_quality/checks.py.
4. Passes check results into the scoring engine (src/data_quality/scoring.py).
5. Generates quality reports for both datasets.
6. Displays the formatted results showing:
   - Quality score & status
   - Passed checks
   - Failed checks / warnings
   - Recommendations
"""
from typing import List, Dict, Any
from src.data_quality.loader import load_dataset
from src.data_quality.checks import (
    check_missing_values,
    check_duplicate_rows,
    check_duplicate_order_ids,
    check_whitespace,
    check_inconsistent_casing,
    check_invalid_dates,
    check_numeric_columns,
    check_invalid_quantities,
    check_outliers,
    check_revenue_consistency,
)
from src.data_quality.scoring import (
    calculate_quality_score,
    generate_quality_report,
    format_report_text,
)


def run_checks_and_score(name: str, file_path: str) -> Dict[str, Any]:
    """
    Loads dataset, runs all 10 checks, generates quality report, and prints findings.
    """
    # 1. Load dataset
    df = load_dataset(file_path)
    
    # 2. Run all 10 quality checks
    check_results = [
        check_missing_values(df),
        check_duplicate_rows(df),
        check_duplicate_order_ids(df, id_column="order_id"),
        check_whitespace(df),
        check_inconsistent_casing(df),
        check_invalid_dates(df, date_columns=["order_date"]),
        check_numeric_columns(df, numeric_columns=["quantity", "unit_price", "revenue"]),
        check_invalid_quantities(df, quantity_column="quantity"),
        check_outliers(df, numeric_columns=["quantity", "unit_price", "revenue"]),
        check_revenue_consistency(
            df,
            quantity_column="quantity",
            unit_price_column="unit_price",
            revenue_column="revenue",
        ),
    ]
    
    # 3. Calculate score & compile report
    score_result = calculate_quality_score(check_results, total_rows=len(df))
    report = generate_quality_report(
        check_results=check_results,
        score_result=score_result,
        total_rows=len(df),
    )
    
    # 4. Verify Mathematical Integrity
    sum_individual_penalties = round(sum(c["penalty"] for c in report["check_details"]), 2)
    
    # Assertion A: Each penalty = round(issue_rate * weight, 2)
    for c in report["check_details"]:
        expected_penalty = round(c["issue_rate"] * c["severity_weight"], 2)
        assert abs(c["penalty"] - expected_penalty) < 1e-4, (
            f"Penalty mismatch in '{c['check_name']}': got {c['penalty']}, expected {expected_penalty}"
        )
    
    # Assertion B: Total penalty == sum of individual penalties
    assert abs(report["total_penalty"] - sum_individual_penalties) < 1e-4, (
        f"Total penalty mismatch: got {report['total_penalty']}, expected sum {sum_individual_penalties}"
    )
    
    # Assertion C: Overall Score == max(0.0, min(100.0, 100.0 - total_penalty))
    expected_score = round(max(0.0, min(100.0, 100.0 - report["total_penalty"])), 2)
    assert abs(report["overall_score"] - expected_score) < 1e-4, (
        f"Score formula mismatch: got {report['overall_score']}, expected {expected_score}"
    )
    
    # Assertion D: Score clamped within [0.0, 100.0]
    assert 0.0 <= report["overall_score"] <= 100.0, (
        f"Score out of bounds [0.0, 100.0]: {report['overall_score']}"
    )

    # 5. Display formatted report
    print(format_report_text(report, title=f"QUALITY REPORT: {name}"))
    print(f" [OK] All mathematical assertions verified for '{name}'.\n")
    return report


def test_score_clamping():
    """
    Verifies that the score is strictly clamped between 0.0 and 100.0
    even under extreme conditions (massive anomalies or zero penalties).
    """
    print("=" * 94)
    print(" VERIFYING SCORE CLAMPING BOUNDARIES [0.0, 100.0]")
    print("=" * 94)

    # Case 1: Catastrophic failure where total penalties exceed 100 (e.g. 100% fail across all high checks)
    catastrophic_results = [
        {"check_name": "Missing Values", "has_issues": True, "total_missing_values": 100},
        {"check_name": "Duplicate 'order_id' Check", "has_issues": True, "duplicate_id_count": 100},
        {"check_name": "Invalid Dates", "has_issues": True, "total_invalid_dates": 100},
        {"check_name": "Revenue Business Rule Consistency", "has_issues": True, "inconsistent_row_count": 100},
    ]
    # Sum of penalties: 4 * 50.0 = 200.0 points deduction
    res_lower = calculate_quality_score(catastrophic_results, total_rows=100)
    print(f" Catastrophic Failure: Deductions = -{res_lower['total_penalty']} pts -> Clamped Score: {res_lower['overall_score']} / 100.0 (Status: {res_lower['status']})")
    assert res_lower["overall_score"] == 0.0, f"Expected 0.0, got {res_lower['overall_score']}"
    assert res_lower["status"] == "Poor"

    # Case 2: Zero issues (perfect dataset)
    zero_results = [
        {"check_name": "Missing Values", "has_issues": False},
        {"check_name": "Exact Duplicate Rows", "has_issues": False},
    ]
    res_upper = calculate_quality_score(zero_results, total_rows=100)
    print(f" Perfect Dataset     : Deductions = -{res_upper['total_penalty']} pts -> Clamped Score: {res_upper['overall_score']} / 100.0 (Status: {res_upper['status']})")
    assert res_upper["overall_score"] == 100.0, f"Expected 100.0, got {res_upper['overall_score']}"
    assert res_upper["status"] == "Excellent"

    print(" [OK] Clamping boundary assertions PASSED.\n")


def test_manual_example():
    """
    Validates the exact small manual example from Step 7:
    - 100 rows
    - 5 missing values (HIGH severity)
    - 2 duplicate IDs (HIGH severity)
    - 1 invalid date (HIGH severity)
    Expected score: 96.0 / 100.0 (Status: Excellent)
    """
    print("=" * 94)
    print(" STEP 7 MANUAL EXAMPLE VALIDATION (100 Rows)")
    print("=" * 94)
    mock_results = [
        {"check_name": "Missing Values", "has_issues": True, "total_missing_values": 5},
        {"check_name": "Exact Duplicate Rows", "has_issues": False, "duplicate_row_count": 0},
        {"check_name": "Duplicate 'order_id' Check", "has_issues": True, "duplicate_id_count": 2},
        {"check_name": "Whitespace Detection", "has_issues": False, "total_whitespace_issues": 0},
        {"check_name": "Inconsistent Capitalization", "has_issues": False, "total_inconsistent_groups": 0},
        {"check_name": "Invalid Dates", "has_issues": True, "total_invalid_dates": 1},
        {"check_name": "Numeric Type Problems", "has_issues": False, "total_problematic_values": 0},
        {"check_name": "Invalid Quantities", "has_issues": False, "negative_quantity_count": 0, "zero_quantity_count": 0},
        {"check_name": "Outlier Detection (IQR)", "has_issues": False, "total_outliers_detected": 0},
        {"check_name": "Revenue Business Rule Consistency", "has_issues": False, "inconsistent_row_count": 0},
    ]

    score_result = calculate_quality_score(mock_results, total_rows=100)
    print(f" Calculated Score   : {score_result['overall_score']} / 100.0")
    print(f" Calculated Status  : {score_result['status']}")
    print(f" Total Deductions   : -{score_result['total_penalty']:.2f} points")
    for c in score_result["check_details"]:
        if c["has_issues"]:
            print(f"  * {c['check_name']} ({c['severity']} severity): {c['issue_count']} issues -> deduction: -{c['penalty']:.2f} pts")

    assert score_result["overall_score"] == 96.0, f"Expected 96.0, got {score_result['overall_score']}"
    assert score_result["status"] == "Excellent", f"Expected Excellent, got {score_result['status']}"
    print(" [OK] Manual example assertion PASSED: 100 - (2.50 + 1.00 + 0.50) = 96.00\n")


def main():
    print("=" * 94)
    print(" DATA QUALITY SCORING ENGINE MATHEMATICAL VERIFICATION")
    print("=" * 94)
    print("Running end-to-end evaluation and mathematical verification on all datasets...\n")
    
    # Test 1: Clean Dataset
    clean_report = run_checks_and_score(
        name="Clean E-Commerce Dataset (CSV)",
        file_path="sample_data/sample_sales.csv",
    )
    
    # Test 2: Dirty Dataset
    dirty_report = run_checks_and_score(
        name="Dirty Dataset with Intentional Anomalies (Excel)",
        file_path="sample_data/sample_dirty_data.xlsx",
    )

    # Test 3: Manual Step 7 Example
    test_manual_example()

    # Test 4: Score Clamping Boundaries
    test_score_clamping()
    
    # Summary Comparison
    print("=" * 94)
    print(" DATASET COMPARISON & VERIFICATION SUMMARY")
    print("=" * 94)
    print(f" Clean Dataset Score : {clean_report['overall_score']} / 100.0  ({clean_report['quality_status']})")
    print(f" Dirty Dataset Score : {dirty_report['overall_score']} / 100.0  ({dirty_report['quality_status']})")
    print(" Mathematical checks : ALL PASSED (Individual penalties, sum of penalties, clamping)")
    print("=" * 94)


if __name__ == "__main__":
    main()

