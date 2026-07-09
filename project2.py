# Project 2: System of Linear Equations and Statistics

import csv

# Task 1: System of linear equations

def read_equations(filename):
    """
    Read coefficients from input file and return matrix A and vector b
    Input format: Each line contains coefficients for one equation: a1,a2,a3,a4,b
    Where: a1*x + a2*y + a3*z + a4*w = b
    """
    A = []  # Coefficient matrix
    b = []  # Constants vector

    try:
        with open(filename, 'r') as file:
            for line in file:
                # Remove whitespace and split by comma
                coefficients = line.strip().split(',')
                if len(coefficients) == 5:  # 4 coefficients + 1 constant
                    # Convert to floats
                    row = [float(coef) for coef in coefficients]
                    A.append(row[:4])  # First 4 are coefficients
                    b.append(row[4])  # Last one is constant
        return A, b
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None, None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None, None


def gaussian_elimination(A, b):
    """
    Solve system of linear equations using Gaussian elimination
    """
    n = len(A)

    # Create augmented matrix
    aug = []
    for i in range(n):
        row = A[i][:]  # Copy coefficients
        row.append(b[i])  # Add constant
        aug.append(row)

    # Forward elimination
    for i in range(n):
        # Partial pivoting: find row with maximum element in current column
        max_row = i
        for j in range(i + 1, n):
            if abs(aug[j][i]) > abs(aug[max_row][i]):
                max_row = j

        # Swap rows
        aug[i], aug[max_row] = aug[max_row], aug[i]

        # Make diagonal element 1 and eliminate below
        pivot = aug[i][i]
        if abs(pivot) < 1e-10:  # Check for singular matrix
            return None  # No unique solution

        # Normalize the pivot row
        for j in range(i, n + 1):
            aug[i][j] /= pivot

        # Eliminate below
        for k in range(i + 1, n):
            factor = aug[k][i]
            for j in range(i, n + 1):
                aug[k][j] -= factor * aug[i][j]

    # Back substitution
    x = [0] * n
    for i in range(n - 1, -1, -1):
        x[i] = aug[i][n]  # Start with the constant
        for j in range(i + 1, n):
            x[i] -= aug[i][j] * x[j]

    return x


def solve_linear_system(filename):
    """
    Main function to solve the system of linear equations
    """
    print("Task 1: Solving System of Linear Equations")

    # Read equations from file
    A, b = read_equations(filename)

    if A is None or b is None:
        return None

    print("System of equations:")
    for i in range(len(A)):
        equation = f"{A[i][0]:.1f}x + {A[i][1]:.1f}y + {A[i][2]:.1f}z + {A[i][3]:.1f}w = {b[i]:.1f}"
        print(f"Equation {i + 1}: {equation}")

    # Solve the system
    solution = gaussian_elimination(A, b)

    if solution is None:
        print("Error: The system does not have a unique solution.")
        print("This could be due to:")
        print("- Infinitely many solutions")
        print("- No solution")
        return None

    print("\nSolutions:")
    print(f"x = {solution[0]:.6f}")
    print(f"y = {solution[1]:.6f}")
    print(f"z = {solution[2]:.6f}")
    print(f"w = {solution[3]:.6f}")

    # Verify solution
    print("\nVerification:")
    for i in range(len(A)):
        calculated = sum(A[i][j] * solution[j] for j in range(4))
        print(f"Equation {i + 1}: LHS = {calculated:.6f}, RHS = {b[i]:.6f}, Difference = {abs(calculated - b[i]):.6e}")

    return solution


# task 2: statistics

def read_boston_housing(filename):
    medv_data = []

    try:
        with open(filename, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                try:
                    # Try different possible column names
                    if 'medv' in row:
                        medv_value = float(row['medv'])
                    elif 'MEDV' in row:
                        medv_value = float(row['MEDV'])
                    else:
                        continue
                    medv_data.append(medv_value)
                except (ValueError, KeyError):
                    continue
        return medv_data
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None


def calculate_max(data):
    """Calculate maximum value"""
    if not data:
        return None
    max_val = data[0]
    for value in data:
        if value > max_val:
            max_val = value
    return max_val


def calculate_min(data):
    """Calculate minimum value"""
    if not data:
        return None
    min_val = data[0]
    for value in data:
        if value < min_val:
            min_val = value
    return min_val


def calculate_range(data):
    """Calculate range (max - min)"""
    max_val = calculate_max(data)
    min_val = calculate_min(data)
    if max_val is None or min_val is None:
        return None
    return max_val - min_val


def calculate_mean(data):
    """Calculate mean (average)"""
    if not data:
        return None
    total = 0
    for value in data:
        total += value
    return total / len(data)


def calculate_mode(data):
    """Calculate mode (most frequent value)"""
    if not data:
        return None

    frequency = {}
    for value in data:
        frequency[value] = frequency.get(value, 0) + 1

    max_freq = 0
    mode_val = None
    for value, freq in frequency.items():
        if freq > max_freq:
            max_freq = freq
            mode_val = value

    return mode_val


def calculate_variance(data):
    """Calculate variance"""
    if not data:
        return None

    mean = calculate_mean(data)
    n = len(data)

    sum_squared_diff = 0
    for value in data:
        sum_squared_diff += (value - mean) ** 2

    return sum_squared_diff / n  # Population variance


def calculate_std_deviation(data):
    """Calculate standard deviation"""
    variance = calculate_variance(data)
    return variance ** 0.5 if variance is not None else None


def sort_data(data):
    """Sort data using bubble sort (implemented manually)"""
    sorted_data = data.copy()
    n = len(sorted_data)

    for i in range(n):
        for j in range(0, n - i - 1):
            if sorted_data[j] > sorted_data[j + 1]:
                sorted_data[j], sorted_data[j + 1] = sorted_data[j + 1], sorted_data[j]

    return sorted_data


def percentile_method1(data, p):
    """
    Method 1: C = (N + 1) * P
    """
    sorted_data = sort_data(data)
    n = len(sorted_data)
    position = (n + 1) * p

    if position.is_integer():
        return sorted_data[int(position) - 1]
    else:
        lower_idx = int(position) - 1
        upper_idx = lower_idx + 1
        fraction = position - int(position)
        return sorted_data[lower_idx] + fraction * (sorted_data[upper_idx] - sorted_data[lower_idx])


def percentile_method2(data, p):
    """
    Method 2: C = N * P + 0.5
    """
    sorted_data = sort_data(data)
    n = len(sorted_data)
    position = n * p + 0.5

    if position.is_integer():
        return sorted_data[int(position) - 1]
    else:
        lower_idx = int(position) - 1
        upper_idx = lower_idx + 1
        fraction = position - int(position)
        return sorted_data[lower_idx] + fraction * (sorted_data[upper_idx] - sorted_data[lower_idx])


def percentile_method3(data, p):
    """
    Method 3: C = N * P
    If C is integer, average of C-th and (C+1)-th values
    Otherwise, round up to nearest integer
    """
    sorted_data = sort_data(data)
    n = len(sorted_data)
    position = n * p

    if position.is_integer():
        idx1 = int(position) - 1
        idx2 = int(position)
        return (sorted_data[idx1] + sorted_data[idx2]) / 2
    else:
        return sorted_data[int(position)]


def analyze_boston_housing():
    """
    Main function for Task 2
    """

    print("Task 2: Statistical Analysis of Boston Housing Data")


    # Read data
    medv_data = read_boston_housing("BostonHousing.csv")

    if medv_data is None:
        return None

    print(f"Number of data points in MEDV column: {len(medv_data)}")

    # Calculate basic statistics
    print("\nBasic Statistics for MEDV Column:")

    max_val = calculate_max(medv_data)
    min_val = calculate_min(medv_data)
    range_val = calculate_range(medv_data)
    mean_val = calculate_mean(medv_data)
    mode_val = calculate_mode(medv_data)
    variance_val = calculate_variance(medv_data)
    std_dev_val = calculate_std_deviation(medv_data)

    print(f"Maximum: {max_val:.6f}")
    print(f"Minimum: {min_val:.6f}")
    print(f"Range: {range_val:.6f}")
    print(f"Mean: {mean_val:.6f}")
    print(f"Mode: {mode_val:.6f}")
    print(f"Variance: {variance_val:.6f}")
    print(f"Standard Deviation: {std_dev_val:.6f}")

    # Calculate percentiles using all three methods
    print("\nPercentiles for MEDV Column:")

    p40_method1 = percentile_method1(medv_data, 0.4)
    p40_method2 = percentile_method2(medv_data, 0.4)
    p40_method3 = percentile_method3(medv_data, 0.4)

    p80_method1 = percentile_method1(medv_data, 0.8)
    p80_method2 = percentile_method2(medv_data, 0.8)
    p80_method3 = percentile_method3(medv_data, 0.8)

    print("40th Percentile:")
    print(f"  Method 1: {p40_method1:.6f}")
    print(f"  Method 2: {p40_method2:.6f}")
    print(f"  Method 3: {p40_method3:.6f}")

    print("\n80th Percentile:")
    print(f"  Method 1: {p80_method1:.6f}")
    print(f"  Method 2: {p80_method2:.6f}")
    print(f"  Method 3: {p80_method3:.6f}")

    return medv_data

if __name__ == "__main__":
    print("Project 2: Linear Equations and Statistics")


    # Run Task 1
    solution_task1 = solve_linear_system("input.txt")

    # Run Task 2
    data_task2 = analyze_boston_housing()
