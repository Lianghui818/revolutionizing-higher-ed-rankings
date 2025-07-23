import os
import sys
import subprocess

# === CONFIGURE YOUR SCRIPT PATHS HERE ===
GET_REFERENCES_PATH = "../llm/t2.py"
AT_PATH = "../llm/get_at.py"
MATCH_ABBR_NAMES_PATH = "../faculty/Scoring/match_abbr_names.py"
CATEGORIZE_AUTHORS_PATH = "categorize_authors.py"
# ========================================

def run_script(script_path, *args):
    """Run a Python script with optional arguments."""
    try:
        command = ["python", script_path]
        if args:
            command.extend(args)
        print(f"Running command: {' '.join(command)}")
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path}: {e}")
        sys.exit(1)

def main():
    if len(sys.argv) != 2:
        print("Usage: python run_all.py <conference_directory>")
        print("Example: python run_all.py NeurIPS_2023")
        sys.exit(1)

    conference_dir = sys.argv[1]
    print(f"Processing conference: {conference_dir}")

    # Run get_references.py on the entire conference directory
    print(f"\nRunning get_references.py on conference {conference_dir}...")
    run_script(GET_REFERENCES_PATH, conference_dir)

    # Run the remaining scripts once
    print("\nRunning at.py...")
    run_script(AT_PATH)

    print("\nRunning match_abbr_names.py...")
    run_script(MATCH_ABBR_NAMES_PATH)

    print("\nRunning categorize_authors.py...")
    run_script(CATEGORIZE_AUTHORS_PATH)

    print("\nProcessing complete for conference:", conference_dir)

if __name__ == "__main__":
    main()