import subprocess
import re
import os
import pandas as pd

def run_ml_model():
    # Run the ML_models.py file and capture the output
    result = subprocess.run(['/home/best/miniconda3/bin/python', 'ML_models.py'], capture_output=True, text=True)
    output = result.stdout

    # Extract accuracy values using regular expressions
    resolution_acc_match = re.search(r'Resolution accuracy: (\d+\.?\d*)%', output)
    fps_acc_match = re.search(r'FPS accuracy: (\d+\.?\d*)%', output)

    # Extract the tables (labels and features) from the output
    resolution_labels_table = re.findall(r'resolution input features:\n(.+?)\n\n', output, re.DOTALL)
    resolution_features_table = re.findall(r'labels input features:\n(.+?)\n\n', output, re.DOTALL)
    fps_labels_table = re.findall(r'FPS input features:\n(.+?)\n\n', output, re.DOTALL)
    fps_features_table = re.findall(r'FPS labels:\n(.+?)\n\n', output, re.DOTALL)

    resolution_acc = float(resolution_acc_match.group(1)) if resolution_acc_match else None
    fps_acc = float(fps_acc_match.group(1)) if fps_acc_match else None

    return resolution_acc, fps_acc, resolution_labels_table, resolution_features_table, fps_labels_table, fps_features_table

def save_to_csv(table_data, filename):
    # Assuming the tables are extracted as lists of lists (rows and columns), you can convert them to a DataFrame
    df = pd.DataFrame([row.split() for row in table_data.split('\n')])
    df.to_csv(filename, index=False)

def main():
    resolution_above_accuracy = False
    fps_above_accuracy = False
    # accuracy = 

    while not (resolution_above_accuracy and fps_above_accuracy):
        resolution_acc, fps_acc, res_labels, res_features, fps_labels, fps_features = run_ml_model()
        # Check if resolution accuracy exceeds accuracy%
        if resolution_acc and resolution_acc > 80 and not resolution_above_accuracy:
            if res_labels and res_features:
                save_to_csv(res_labels[0], 'resolution_labels.csv')
                save_to_csv(res_features[0], 'resolution_features.csv')
            print(f"Resolution accuracy {resolution_acc}%: Data saved.")
            resolution_above_accuracy = True

        # Check if FPS accuracy exceeds accuracy%
        if fps_acc and fps_acc > 80 and not fps_above_accuracy:
            if fps_labels and fps_features:
                save_to_csv(fps_labels[0], 'fps_labels.csv')
                save_to_csv(fps_features[0], 'fps_features.csv')
            print(f"FPS accuracy {fps_acc}%: Data saved.")
            fps_above_accuracy = True

if __name__ == '__main__':
    main()
    os.system("/home/best/miniconda3/bin/python /home/best/Desktop/EEE4022S/scripts/overall_script.py")
