import subprocess
import sys
import time

def run_script(script_path):
    """
    Runs a Python script and handles any errors
    
    Args:
        script_path (str): Path to the Python script to run
    Returns:
        bool: True if successful, False if failed
    """
    try:
        print(f"\nRunning: {script_path}")
        print("-" * 50)
        
        # Run the script using the same Python interpreter
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, 
                              text=True, 
                              check=True)
        
        # Print the output
        if result.stdout:
            print("Output:")
            print(result.stdout)
            
        print(f"Successfully completed: {script_path}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path}")
        print(f"Error message: {e.stderr}")
        return False
    except Exception as e:
        print(f"Unexpected error running {script_path}: {str(e)}")
        return False

def main():
    # List of scripts to run in order
    scripts = [
        "preprocessing_solar2.py",  
        "solar_analysis.py",
        "solar_mapping.py"
    ]
    
    start_time = time.time()
    success_count = 0
    
    print("Starting script execution sequence...")
    
    # Run each script in succession
    for script in scripts:
        if run_script(script):
            success_count += 1
        else:
            print(f"\nScript failed: {script}")
            user_input = input("Continue with remaining scripts? (y/n): ")
            if user_input.lower() != 'y':
                print("Execution sequence aborted.")
                break
    
    # Print summary
    end_time = time.time()
    duration = end_time - start_time
    
    print("\nExecution Summary")
    print("-" * 50)
    print(f"Total scripts: {len(scripts)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(scripts) - success_count}")
    print(f"Total duration: {duration:.2f} seconds")

if __name__ == "__main__":
    main()