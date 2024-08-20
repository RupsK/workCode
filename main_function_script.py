import subprocess
import os
import time
import json

def run_script(script_path, *args):
    try:
        result = subprocess.run(['python', script_path, *args], capture_output=True, text=True)
        print(f"Output of {script_path}:")
        print(result.stdout)
        if result.stderr:
            print(f"Error output of {script_path}:")
            print(result.stderr)
        return result.returncode
    except Exception as e:
        print(f"Exception while running script {script_path}: {e}")
        return 1

def wait_for_folder(folder_path, timeout=60):
    start_time = time.time()
    while not os.path.exists(folder_path):
        if time.time() - start_time > timeout:
            print(f"Timeout: Folder {folder_path} not found after {timeout} seconds.")
            return False
        time.sleep(1)
    return True

def read_rc_enforced_pole_flag():
    try:
        with open('rc_enforced_pole_flag.txt', 'r') as f:
            flag = f.read().strip()
            return flag.lower() == 'true'
    except FileNotFoundError:
        return False

def read_rc_pole_flag():
    try:
        with open('rc_pole_flag.txt', 'r') as f:
            flag = f.read().strip()
            return flag.lower() == 'true'
    except FileNotFoundError:
        return False
    
def update_json_file(json_file_path, detection_result):
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r', encoding='utf-8') as f:
            result_data = json.load(f)
    else:
        result_data = {}

    if not isinstance(result_data, dict):
        result_data = {}

    if 'details' not in result_data:
        result_data['details'] = []

    result_data['details'].append(detection_result)

    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=4)

# Paths to scripts and files
first_script_path = "C:/Rupali Shinde/Source code/save_all_cropped_view_bestResult.py"
second_script_path = "C:/Rupali Shinde/Source code/input_folder_path.py"
third_script_path = "C:/Rupali Shinde/Source code/final_RC_2_condition_check.py"
image_dir=  "C:/Users/h4tec/Documents/KakaoTalk Downloads/images_PCPole/images_121201013_20240604133901_1/"             #"C:/Users/h4tec/Desktop/jejuPC41/"
json_file = "C:/Users/h4tec/Documents/KakaoTalk Downloads/images_PCPole/images_121201013_20240604133901_1/output.json"

output_folder = os.path.join(os.path.dirname(first_script_path), 'output/views/cropped_view')
output_folder2 = os.path.join(os.path.dirname(first_script_path), 'output/cropped')

# Run the first script
first_result = run_script(first_script_path, image_dir, json_file)
if first_result == 0:
    # Check if the output folder is created
    if wait_for_folder(output_folder):
        # Run the second script if the output folder exists
        second_result = run_script(second_script_path, output_folder)
        if second_result == 0:
            print("First and second scripts ran successfully.")
            # Check if RC enforced pole was detected
            if not read_rc_enforced_pole_flag():
                # Run the third script if the second script succeeds and RC enforced pole is not detected
                third_result = run_script(third_script_path, output_folder2)
                if third_result == 0:
                    print("All three scripts ran successfully.")
                    if read_rc_pole_flag() ==0:
                        # Print "Unknown pole category" if RC pole not detected
                        print("Unknown pole category")
                    
                else:
                    print("Third script failed.")
            else:
                print("RC enforced pole detected. Skipping the third script.")
        else:
            print("Second script failed.")
    else:
        print("Required output folder not found. Exiting.")
else:
    print("First script failed. Exiting.")
