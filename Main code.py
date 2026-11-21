"""
Student Academic Performance Analytics (HCL Project 2)
Uses only Python, NumPy and Pandas. No charts.
Run from the repository folder:  python src/analysis.py
or from the src folder:         python analysis.py
"""


# ======================================================================
# Step 1: Import libraries and set the file paths
# ======================================================================
import os
import numpy as np
import pandas as pd

# Show all columns when printing tables
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

# Find the data file relative to this script, with support for common layouts.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILENAME = "02_student_performance_updated_02_student_performance.csv"

candidate_paths = [
    os.path.join(SCRIPT_DIR, DATA_FILENAME),
    os.path.join(SCRIPT_DIR, "data", DATA_FILENAME),
    os.path.join(SCRIPT_DIR, "data", "02_student_performance.csv"),
    os.path.join(os.getcwd(), DATA_FILENAME),
    os.path.join(os.getcwd(), "data", "02_student_performance.csv"),
]

DATA_PATH = next((path for path in candidate_paths if os.path.isfile(path)), None)
if DATA_PATH is None:
    raise FileNotFoundError(
        "Could not find the student performance CSV. Place "
        "02_student_performance_updated_02_student_performance.csv next to "
        "analysis.py or in a data/ folder."
    )

# Folder where result files will be saved (the original CSV is never changed)
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# The three subject columns used for total and average marks
SUBJECTS = ["Python_Marks", "Maths_Marks", "Physics_Marks"]

print("Data file:", DATA_PATH)
print("Subject columns:", SUBJECTS)


# ======================================================================
# Step 2: Load the dataset (Task 1)
# ======================================================================
raw_df = pd.read_csv(DATA_PATH)   # original data - we never change this
df = raw_df.copy()                # working copy used for cleaning and analysis

print("File format: CSV")
print("Number of rows   :", df.shape[0])
print("Number of columns:", df.shape[1])


# ======================================================================
# Step 3: First and last records
# ======================================================================
print("First 5 records:")
print(df.head())
print()
print("Last 5 records:")
print(df.tail())


# ======================================================================
# Step 4: Shape, column names, data types and summary
# ======================================================================
print("Shape (rows, columns):", df.shape)
print()
print("Column names:")
print(list(df.columns))
print()
df.info()
print()
print("Summary of number columns:")
print(df.describe().round(2))


# ======================================================================
# Step 5: Check which calculated columns already exist
# ======================================================================
calculated_columns = ["Total_Marks", "Max_Marks", "Average_Marks", "Percentage",
                      "Grade", "At_Risk", "Attendance_Band", "Total_Assignments"]

for col in calculated_columns:
    if col in df.columns:
        print(col, "-> exists in the file")
    else:
        print(col, "-> not in the file")

# Is the stored Total_Marks equal to the sum of the three subject marks?
calculated_total = df[SUBJECTS].sum(axis=1)
if "Total_Marks" in df.columns:
    mismatch_count = (calculated_total != df["Total_Marks"]).sum()
    print()
    print("Rows where Total_Marks does not match the subject marks:", mismatch_count)
else:
    print()
    print("Rows where Total_Marks does not match the subject marks: Total_Marks not in file")


# ======================================================================
# Step 6: Check missing values (Task 2)
# ======================================================================
missing_count = df.isna().sum()
missing_percent = (df.isna().mean() * 100).round(2)

missing_table = pd.DataFrame({"Missing": missing_count, "Percent": missing_percent})
print(missing_table)
print()
print("Total missing cells:", int(missing_count.sum()))


# ======================================================================
# Step 7: Check and remove duplicate Student IDs (Task 2)
# ======================================================================
repeated_id_count = df["Student_ID"].duplicated().sum()
exact_copy_count = df.duplicated().sum()

print("Rows with a repeated Student_ID:", repeated_id_count)
print("Rows that are exact copies of an earlier row:", exact_copy_count)

repeated_rows = df[df["Student_ID"].duplicated(keep=False)].sort_values("Student_ID")
print()
print("All rows that share a Student_ID:")
print(repeated_rows[["Student_ID", "Department", "Attendance_Percent",
                     "Python_Marks", "Maths_Marks", "Physics_Marks"]])

if repeated_id_count == exact_copy_count:
    print()
    print("Every repeated ID is an exact copy, so dropping the later copies loses no information.")

rows_before = len(df)
df = df.drop_duplicates(subset="Student_ID", keep="first").reset_index(drop=True)
print()
print("Rows before:", rows_before, "| Rows after:", len(df), "| Removed:", rows_before - len(df))


# ======================================================================
# Step 8: Handle missing attendance values (Task 3)
# ======================================================================
missing_att_before = df["Attendance_Percent"].isna().sum()

print("Missing attendance values before:", missing_att_before)
print()
print("Missing attendance by department:")
print(df[df["Attendance_Percent"].isna()]["Department"].value_counts())

# Median attendance of each department (one value per row)
dept_median_attendance = df.groupby("Department")["Attendance_Percent"].transform("median")
df["Attendance_Percent"] = df["Attendance_Percent"].fillna(dept_median_attendance)
df["Attendance_Percent"] = df["Attendance_Percent"].fillna(df["Attendance_Percent"].median())

print()
print("Median attendance used for filling:")
print(df.groupby("Department")["Attendance_Percent"].median())
print()
print("Missing attendance values after:", df["Attendance_Percent"].isna().sum())


# ======================================================================
# Step 9: Check that subject marks are between 0 and 100 (Task 4)
# ======================================================================
def count_out_of_range(data, columns, low=0, high=100):
    """Count the values outside the range low-high in each column."""
    result = {}
    for col in columns:
        wrong = (data[col] < low) | (data[col] > high)
        result[col] = int(wrong.sum())
    return result

print("Marks outside 0-100 (before fixing):")
print(count_out_of_range(df, SUBJECTS))

for col in SUBJECTS:
    wrong_rows = df[(df[col] < 0) | (df[col] > 100)]
    if len(wrong_rows) > 0:
        print()
        print("Wrong values in", col)
        print(wrong_rows[["Student_ID", "Department", col]])


# ======================================================================
# Step 10: Fix the invalid marks (Task 4)
# ======================================================================
for col in SUBJECTS:
    is_wrong = (df[col] < 0) | (df[col] > 100)
    df.loc[is_wrong, col] = np.nan                        # remove the impossible value
    dept_median_marks = df.groupby("Department")[col].transform("median")
    df[col] = df[col].fillna(dept_median_marks)           # fill with department median
    df[col] = df[col].fillna(df[col].median())            # final fallback for empty departments

print("Marks outside 0-100 (after fixing):")
print(count_out_of_range(df, SUBJECTS))
print("Missing subject marks after fixing:", int(df[SUBJECTS].isna().sum().sum()))


# ======================================================================
# Step 11: Calculate total and average marks (Task 5)
# ======================================================================
df["Total_Marks"] = df[SUBJECTS].sum(axis=1)
df["Max_Marks"] = 100 * len(SUBJECTS)
df["Average_Marks"] = (df["Total_Marks"] / len(SUBJECTS)).round(2)
df["Percentage"] = (df["Total_Marks"] / df["Max_Marks"] * 100).round(2)

print(df[["Student_ID", "Python_Marks", "Maths_Marks", "Physics_Marks",
          "Total_Marks", "Max_Marks", "Average_Marks", "Percentage"]].head(10))
print()
print("Highest average:", df["Average_Marks"].max(), "| Lowest average:", df["Average_Marks"].min())


# ======================================================================
# Step 12: Create the Grade column (Task 6)
# ======================================================================
grade_bins = [0, 40, 50, 60, 75, 90, 100.01]
grade_labels = ["F", "D", "C", "B", "A", "A+"]

df["Grade"] = pd.cut(df["Average_Marks"], bins=grade_bins, labels=grade_labels, right=False)

grade_counts = df["Grade"].value_counts().reindex(grade_labels)
grade_table = pd.DataFrame({"Students": grade_counts,
                            "Percent": (grade_counts / len(df) * 100).round(1)})
print("Grade distribution:")
print(grade_table)
print()
print("Students without a grade:", df["Grade"].isna().sum())


# ======================================================================
# Step 13: Department-wise average marks (Task 7)
# ======================================================================
dept_summary = df.groupby("Department").agg(
    Students=("Student_ID", "count"),
    Average_Marks=("Average_Marks", "mean"),
    Median_Marks=("Average_Marks", "median"),
    Mean_Attendance=("Attendance_Percent", "mean"),
    Mean_Study_Hours=("Study_Hours_Per_Day", "mean"),
).round(2)

dept_summary = dept_summary.sort_values("Average_Marks", ascending=False)
print(dept_summary)
print()
print("Total students:", dept_summary["Students"].sum())


# ======================================================================
# Step 14: Top 10 students (Task 8)
# ======================================================================
top10 = df.sort_values(["Average_Marks", "Student_ID"], ascending=[False, True]).head(10)
print(top10[["Student_ID", "Department", "Average_Marks", "Grade"]].to_string(index=False))

# Are there other students with the same average as the 10th student?
tenth_average = top10["Average_Marks"].iloc[-1]
students_with_same = (df["Average_Marks"] == tenth_average).sum()
print()
print("Average of the 10th student:", tenth_average)
print("Students with exactly this average:", students_with_same)


# ======================================================================
# Step 15: Students below 40% average (Task 9)
# ======================================================================
df["At_Risk"] = np.where(df["Average_Marks"] < 40, "Yes", "No")

at_risk = df[df["At_Risk"] == "Yes"].sort_values("Average_Marks")
at_risk_count = len(at_risk)
at_risk_percent = at_risk_count / len(df) * 100

print("Students below 40% average:", at_risk_count, "out of", len(df), "(" + str(round(at_risk_percent, 1)) + "%)")
print()
print(at_risk[["Student_ID", "Department", "Attendance_Percent",
               "Study_Hours_Per_Day", "Average_Marks"]].to_string(index=False))
print()
print("At-risk students by department:")
print(at_risk["Department"].value_counts())


# ======================================================================
# Step 16: Compare marks across attendance bands (Task 10)
# ======================================================================
att_bins = [0, 60, 75, 90, 100.01]
att_labels = ["<60", "60-74.99", "75-89.99", "90+"]

df["Attendance_Band"] = pd.cut(df["Attendance_Percent"], bins=att_bins, labels=att_labels, right=False)

att_summary = df.groupby("Attendance_Band", observed=True)["Average_Marks"].agg(
    Students="count", Mean="mean", Median="median").round(2)
print(att_summary)
print()
print("Students in each attendance band by At_Risk:")
print(pd.crosstab(df["Attendance_Band"], df["At_Risk"]))


# ======================================================================
# Step 17: Study hours and average marks (Task 11)
# ======================================================================
def describe_correlation(r):
    """Turn a correlation value into a short description."""
    size = abs(r)
    if size < 0.1:
        strength = "very weak or no"
    elif size < 0.3:
        strength = "weak"
    elif size < 0.5:
        strength = "moderate"
    else:
        strength = "strong"

    if r > 0:
        direction = "positive"
    elif r < 0:
        direction = "negative"
    else:
        direction = ""
    return (strength + " " + direction).strip()

correlation_data = df[["Study_Hours_Per_Day", "Average_Marks"]].dropna()
study_hours = correlation_data["Study_Hours_Per_Day"].to_numpy()
avg_marks = correlation_data["Average_Marks"].to_numpy()

r_study = np.corrcoef(study_hours, avg_marks)[0, 1]
slope, intercept = np.polyfit(study_hours, avg_marks, 1)

print("Correlation (r) between study hours and average marks:", round(r_study, 3))
print("Meaning:", describe_correlation(r_study), "relationship")
print("Best-fit line: Average marks = ", round(slope, 2), "x study hours +", round(intercept, 2))
print("So each extra study hour per day is ASSOCIATED with about", round(slope, 2), "more marks.")
print("(This shows association, not cause and effect.)")

# Compare marks across study-hour bands
study_bins = [0, 2, 4, 6, 8.01]
study_labels = ["<2", "2-3.99", "4-5.99", "6+"]
study_band = pd.cut(df["Study_Hours_Per_Day"], bins=study_bins, labels=study_labels, right=False)

print()
print("Students outside the study-hour bands:", study_band.isna().sum())
study_summary = df.groupby(study_band, observed=True)["Average_Marks"].agg(
    Students="count", Mean="mean", Median="median").round(2)
print(study_summary)


# ======================================================================
# Step 18: Compare students with and without internet access (Task 12)
# ======================================================================
net_summary = df.groupby("Internet_Access")["Average_Marks"].agg(
    Students="count", Mean="mean", Median="median", Std="std").round(2)
print(net_summary)

net_risk_percent = (pd.crosstab(df["Internet_Access"], df["At_Risk"], normalize="index") * 100).round(1)
print()
print("Percent of students At_Risk (Yes/No) inside each internet group:")
print(net_risk_percent)

mean_with_net = net_summary.loc["Yes", "Mean"]
mean_without_net = net_summary.loc["No", "Mean"]
print()
print("Difference in mean marks (Yes minus No):", round(mean_with_net - mean_without_net, 2))


# ======================================================================
# Step 19: NumPy statistics for Python_Marks (Task 13)
# ======================================================================
python_marks = df["Python_Marks"].to_numpy()

py_mean = np.mean(python_marks)
py_median = np.median(python_marks)
py_std = np.std(python_marks, ddof=1)
q1, q2, q3 = np.percentile(python_marks, [25, 50, 75])

print("Number of values:", len(python_marks))
print("Mean               :", round(py_mean, 2))
print("Median             :", py_median)
print("Standard deviation :", round(py_std, 2))
print("Q1 (25th percentile):", q1)
print("Q2 (50th percentile):", q2)
print("Q3 (75th percentile):", q3)
print("Minimum / Maximum  :", python_marks.min(), "/", python_marks.max())


# ======================================================================
# Step 20: Find outliers in Python_Marks with the IQR method (Task 14)
# ======================================================================
iqr = q3 - q1
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr

is_outlier = (python_marks < lower_fence) | (python_marks > upper_fence)
outliers = df[is_outlier]
outlier_count = len(outliers)

print("IQR:", iqr)
print("Lower fence:", lower_fence, "| Upper fence:", upper_fence)
print("Number of outliers:", outlier_count)
print()
print(outliers[["Student_ID", "Department", "Python_Marks", "Average_Marks"]].to_string(index=False))


# ======================================================================
# Step 21: Five recommendations (Task 15)
# ======================================================================
# Values taken from the earlier results
low_band = att_summary.index[0]
high_band = att_summary.index[-1]
low_band_mean = att_summary["Mean"].iloc[0]
high_band_mean = att_summary["Mean"].iloc[-1]
warning_limit = att_bins[2]          # 75 (the start of the third attendance band)

low_study_band = study_summary.index[0]
low_study_mean = study_summary["Mean"].iloc[0]
high_study_band = study_summary.index[-1]
high_study_mean = study_summary["Mean"].iloc[-1]
high_study_count = study_summary["Students"].iloc[-1]

best_dept = dept_summary["Average_Marks"].idxmax()
worst_dept = dept_summary["Average_Marks"].idxmin()
best_dept_mean = dept_summary["Average_Marks"].max()
worst_dept_mean = dept_summary["Average_Marks"].min()

rec1 = ("1. Early-warning list: " + str(at_risk_count) + " students (" + str(round(at_risk_percent, 1)) +
        "%) have an average below 40. Give each of them a mentor and review progress every two weeks. "
        "Also check the " + str(outlier_count) + " Python_Marks outliers one by one to see whether they "
        "are real learning problems or data errors.")

rec2 = ("2. Attendance alerts: students in the " + str(low_band) + " attendance band average " +
        str(low_band_mean) + " marks, compared with " + str(high_band_mean) + " in the " + str(high_band) +
        " band. Send an alert to students whose attendance falls below " + str(warning_limit) + "%.")

rec3 = ("3. Study support: the correlation between study hours and average marks is r = " +
        str(round(r_study, 2)) + " (" + describe_correlation(r_study) + "). Students studying " + str(low_study_band) +
        " hours per day average " + str(low_study_mean) + " marks versus " + str(high_study_mean) + " for " +
        str(high_study_band) + " hours (only " + str(high_study_count) + " students in that band). "
        "Offer study-planning sessions to the low-study group. "
        "This is an association, not proof of cause.")

rec4 = ("4. Department focus: " + str(worst_dept) + " has the lowest average marks (" + str(worst_dept_mean) +
        ") and " + str(best_dept) + " the highest (" + str(best_dept_mean) +
        "). Arrange extra tutorials and peer-learning in " + str(worst_dept) + ".")

if mean_without_net < mean_with_net:
    rec5 = ("5. Digital access: students without internet access average " + str(mean_without_net) +
            " marks versus " + str(mean_with_net) + " with internet. Provide campus lab hours and "
            "offline learning material for them. Check the group sizes first, because the groups are unequal.")
else:
    rec5 = ("5. Digital access: students without internet access do not average fewer marks (" +
            str(mean_without_net) + " versus " + str(mean_with_net) + " with internet), so no special "
            "internet support is suggested by this data.")

for rec in [rec1, rec2, rec3, rec4, rec5]:
    print(rec)
    print()


# ======================================================================
# Step 22: Save the results to files
# ======================================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

df.to_csv(os.path.join(OUTPUT_DIR, "cleaned_student_data.csv"), index=False)
at_risk.to_csv(os.path.join(OUTPUT_DIR, "at_risk_students.csv"), index=False)
dept_summary.to_csv(os.path.join(OUTPUT_DIR, "department_summary.csv"))
top10.to_csv(os.path.join(OUTPUT_DIR, "top10_students.csv"), index=False)

print("Saved files in:", OUTPUT_DIR)
for name in sorted(os.listdir(OUTPUT_DIR)):
    print(" -", name)

# Confirm the original data was not changed
print()
print("Original data still has", raw_df.shape[0], "rows and", raw_df.shape[1], "columns.")


# ======================================================================
# Step 23: Summary of cleaning decisions and limitations
# ======================================================================
