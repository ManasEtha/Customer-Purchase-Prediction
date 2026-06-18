import pandas as pd
import numpy as np

def generate_black_friday_data(filename, num_rows, is_train=True):
    np.random.seed(42 if is_train else 24)
    
    data = {
        'User_ID': np.random.randint(1000001, 1005000, size=num_rows),
        'Product_ID': [f"P00{np.random.randint(1000, 5000)}" for _ in range(num_rows)],
        'Gender': np.random.choice(['M', 'F'], size=num_rows, p=[0.75, 0.25]),
        'Age': np.random.choice(['0-17', '18-25', '26-35', '36-45', '46-50', '51-55', '55+'], size=num_rows),
        'Occupation': np.random.randint(0, 21, size=num_rows),
        'City_Category': np.random.choice(['A', 'B', 'C'], size=num_rows, p=[0.3, 0.4, 0.3]),
        'Stay_In_Current_City_Years': np.random.choice(['0', '1', '2', '3', '4+'], size=num_rows),
        'Marital_Status': np.random.choice([0, 1], size=num_rows, p=[0.6, 0.4]),
        'Product_Category_1': np.random.randint(1, 20, size=num_rows),
        'Product_Category_2': np.random.choice([np.nan, 2, 5, 8, 14, 16], size=num_rows, p=[0.4, 0.2, 0.1, 0.1, 0.1, 0.1]),
        'Product_Category_3': np.random.choice([np.nan, 3, 4, 9, 15, 17], size=num_rows, p=[0.7, 0.1, 0.05, 0.05, 0.05, 0.05])
    }
    
    df = pd.DataFrame(data)
    
    if is_train:
        # Generate base purchase amount and add slight noise based on demographics
        base_purchase = np.random.normal(9000, 3000, size=num_rows)
        # Men and specific product categories tend to spend slightly more in this dataset logic
        gender_modifier = np.where(df['Gender'] == 'M', 1200, 0)
        cat_modifier = df['Product_Category_1'] * 150
        df['Purchase'] = np.abs(base_purchase + gender_modifier + cat_modifier).astype(int)
        
    df.to_csv(filename, index=False)
    print(f"Generated {filename} with {num_rows} rows.")

# Generate datasets
generate_black_friday_data('blackfriday_train.csv', num_rows=10000, is_train=True)
generate_black_friday_data('blackfriday_test.csv', num_rows=2000, is_train=False)