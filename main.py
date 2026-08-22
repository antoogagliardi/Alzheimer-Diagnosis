import subprocess
import os
import sys

EXECUTION_VAR = True

if __name__ == "__main__":
    print("== Alzheimer Diagnosis Project ==")
    # Add some directories to the project folder
    os.makedirs("ckpt", exist_ok=True)
    os.makedirs("confusion_matrices", exist_ok=True)
    os.makedirs("data_split", exist_ok=True)
    os.makedirs("wandb", exist_ok=True)
    
    while(EXECUTION_VAR):
        print("1. Generate Data")
        print("2. Train Model")
        print("3. Exit")
        choice = int(input("Enter your choice: "))

        if choice == 1:
            subprocess.run(["python", "scripts/generate_data.py"])
        elif choice == 2:
            subprocess.run(["python", "scripts/train.py"])
        elif choice == 3:
            EXECUTION_VAR = False
        else:
            print("Invalid user choice. Please try again.")
        
        subprocess.run(["clear"])  # Clear the terminal screen for better readability

    print("Exiting...\n" \
          "End of the program.")
    sys.exit(0)
