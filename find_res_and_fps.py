import subprocess
import re
import pandas as pd
import os

def run_ml_model():
    # Run the ML_models.py file and capture the output
    result = subprocess.run(['python', 'ML_models.py'], capture_output=True, text=True)
    output = result.stdout

    # Extract accuracy values using regular expressions
    resolution_acc_match = re.search(r'Resolution accuracy: (\d+\.?\d*)%', output)
    fps_acc_match = re.search(r'FPS accuracy: (\d+\.?\d*)%', output)

    # Extract the tables (labels and features) from the output
    # Assuming the tables are in a format that can be parsed from the output, you may need to adjust this part
    resolution_labels_table = re.findall(r'Resolution labels table:\n(.+?)\n\n', output, re.DOTALL)
    resolution_features_table = re.findall(r'Resolution features table:\n(.+?)\n\n', output, re.DOTALL)
    fps_labels_table = re.findall(r'FPS labels table:\n(.+?)\n\n', output, re.DOTALL)
    fps_features_table = re.findall(r'FPS features table:\n(.+?)\n\n', output, re.DOTALL)

    resolution_acc = float(resolution_acc_match.group(1)) if resolution_acc_match else None
    fps_acc = float(fps_acc_match.group(1)) if fps_acc_match else None

    return resolution_acc, fps_acc, resolution_labels_table, resolution_features_table, fps_labels_table, fps_features_table

def save_to_csv(table_data, filename):
    # Assuming the tables are extracted as lists of lists (rows and columns), you can convert them to a DataFrame
    df = pd.DataFrame([row.split() for row in table_data.split('\n')])
    df.to_csv(filename, index=False)

def main():
    while True:
        resolution_acc, fps_acc, res_labels, res_features, fps_labels, fps_features = run_ml_model()

        # Check if resolution accuracy exceeds 90%
        if resolution_acc > 90:
            if res_labels and res_features:
                save_to_csv(res_labels[0], 'resolution_labels.csv')
                save_to_csv(res_features[0], 'resolution_features.csv')
            print(f"Resolution accuracy {resolution_acc}%: Data saved.")
            break

        # Check if FPS accuracy exceeds 90%
        if fps_acc > 90:
            if fps_labels and fps_features:
                save_to_csv(fps_labels[0], 'fps_labels.csv')
                save_to_csv(fps_features[0], 'fps_features.csv')
            print(f"FPS accuracy {fps_acc}%: Data saved.")
            break

if __name__ == '__main__':
    main()
