import time
import os
import glob
import msvcrt  
from Atom_defringing import ImageDefringing
import csv
import numpy as np

# --- CONFIGURATION ---
WATCH_FOLDER = r"D:\Input files"
FILE_EXTENSION = "*.sif"
OUTPUT_FILE=r"D:\Output files\shot_statistics.csv"


def get_latest_file(folder, extension):
    """
    Definition-Finds the most recently created file in the folder.
    Input-folder to be tracked, extension of the file
    Output-Path of most recently created file, or None if folder is empty
    """
    search_pattern = os.path.join(folder, extension)
    list_of_files = glob.glob(search_pattern)
    if not list_of_files:
        return None
    return max(list_of_files, key=os.path.getctime)


def is_file_ready(filepath, wait=0.3):
    """
    Definition-Checks if next file is ready
    Input-path of the file, wait time
    Output- Binary output whether file is ready or not
    """
    try:
        #If filesize remains constant, the file is ready
        size1 = os.path.getsize(filepath)
        time.sleep(wait)
        size2 = os.path.getsize(filepath)
        return size1 == size2
    except OSError:
        return False

def log_display(statdict):
    """
    Definition-Writes the results of the cleaning onto a CSV file
    Input-dictionary containing the results
    Output- writes onto the csv file
    """
    file_exists=os.path.isfile(OUTPUT_FILE)
    with open(OUTPUT_FILE, mode='a', newline='') as f:
        #Writing the contents of the dictionary onto the CSV file
        writer=csv.DictWriter(f, fieldnames=statdict.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(statdict)



def start_watchdog():
    print("Starting processor")
    processor = ImageDefringing(mode='auto')  #Defining the main class for denoising
    
    print(f"Waiting for new files in {WATCH_FOLDER}...")
    # Instructions to stop the program
    print("-" * 50)
    print(" >>> PRESS 'q' AT ANY TIME TO STOP THE SCRIPT <<< ")
    print("-" * 50)

    last_processed_file = None

    try:
        while True:
            
            # 1. Check if a key has been pressed this loop
            if msvcrt.kbhit():
                # Read the key that was pressed
                key = msvcrt.getch()
                
                # Check if it is 'q' or 'Q' (using byte strings)
                if key.lower() == b'q':
                    print("\n[USER HALT] 'q' key detected. Shutting down processor")
                    break  # This gracefully exits the while loop

            # 2. Proceed with normal file checking
            latest_file = get_latest_file(WATCH_FOLDER, FILE_EXTENSION)

            if latest_file and latest_file != last_processed_file:
                print(f"\n[NEW SHOT DETECTED]: {os.path.basename(latest_file)}")

                # Wait until Solis finishes writing
                if not is_file_ready(latest_file):
                    print("File not ready yet, waiting...")
                    continue                    

                try:
                    clean_od,current_stats = processor.atom_denoising(latest_file)
                    if current_stats is None:
                        print(f"WARNING: Stats unavailable for {os.path.basename(latest_file)}")
                        last_processed_file = latest_file
                        continue
                    current_stats["Shot_Name"] = os.path.basename(latest_file)
                    current_stats["Time"] = time.strftime("%H:%M:%S")
                    
                    
                    # 3. Save to the CSV instantly
                    log_display(current_stats)
                    print(f"Stats saved to CSV. ")
                    print("Processing complete. Waiting for next shot...")
                    last_processed_file = latest_file

                except Exception as e:
                    print(f"FAILED TO PROCESS {latest_file}. Error: {e}")
                    last_processed_file = latest_file

            time.sleep(1)                      

    except KeyboardInterrupt:
        print("\n[SYSTEM HALT] Watchdog stopped via Ctrl+C.") 
        
    print("Analysis session ended.")

#Running the main watchdog function
if __name__ == "__main__":
    if not os.path.exists(WATCH_FOLDER):
        os.makedirs(WATCH_FOLDER)
    output_dir = os.path.dirname(OUTPUT_FILE)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)    
    start_watchdog()