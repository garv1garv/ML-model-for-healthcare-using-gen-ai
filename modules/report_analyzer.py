import os
import cv2
import pytesseract
import openai

def analyze_report(image_path):
    # Check if the image exists
    if not os.path.exists(image_path):
        print("Image file not found:", image_path)
        return None

    # Read the image using OpenCV
    img = cv2.imread(image_path)
    # Extract text from the image using pytesseract
    extracted_text = pytesseract.image_to_string(img)
    print("Extracted Report Text:")
    print(extracted_text)

    # Compose a prompt for the generative AI model
    prompt = (
        "You are a helpful medical diagnostic assistant. Based on the following report text, "
        "analyze the information and generate a detailed recovery plan with suggested next steps, "
        "potential diagnostics, and treatment recommendations.\n\n"
        "Report Text:\n" + extracted_text
    )

    # Set up the OpenAI API key from environment variable
    openai.api_key = os.getenv("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")
    if not openai.api_key:
        print("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        return None

    try:
        # Make a call to the generative AI model (e.g., GPT-4)
        response = openai.ChatCompletion.create(
            model="gpt-4",  # or "gpt-3.5-turbo" if desired
            messages=[
                {"role": "system", "content": "You are a helpful medical diagnostic assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300  # adjust as needed
        )
        recovery_plan = response['choices'][0]['message']['content']
    except Exception as e:
        print("Error during AI model call:", e)
        return None

    return recovery_plan

# Standalone test
if __name__ == '__main__':
    # Adjust the path to your sample report image
    sample_report = os.path.join(os.path.dirname(__file__), "reports", "sample_report.jpg")
    plan = analyze_report(sample_report)
    if plan:
        print("Recovery Plan:")
        print(plan)
    else:
        print("Failed to generate a recovery plan.")
