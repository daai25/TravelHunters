import subprocess

def run_script_with_input(script_path, input_text):
    """
    Runs a Python script and sends input_text to its stdin.
    Returns the stdout output as a string.
    """
    result = subprocess.run(
        ["python", script_path],
        input=input_text,    # pass string directly, no encode()
        capture_output=True,
        text=True,           # important: treat input/output as text (str)
        check=False
    )
    return result.stdout, result.stderr


if __name__ == "__main__":
    image_path = input("File path: ").strip()
    user_text = input("Enter your text query: ").strip()


    predictor_path = "./predictor.py"
    recommender_path = "./hotel_recommender.py"

    print("\nRunning predictor.py...")
    pred_out, pred_err = run_script_with_input(predictor_path, image_path)
    if pred_err:
        print("Predictor STDERR:", pred_err)
    
        # Parse the output
    lines = pred_out.strip().splitlines()
    city = lines[0].strip()
    confidence = float(lines[1].strip())

    if confidence > 0.85:
        user_text = user_text + "I would like to visit {city} with a {confidence} level of confidence."
    
    
    print("\nRunning hotel_recommender.py...")
    rec_out, rec_err = run_script_with_input(recommender_path, user_text)
    if rec_err:
        print("Hotel Recommender STDERR:", rec_err)
    print("Hotel Recommender Output:")
    print(rec_out)
