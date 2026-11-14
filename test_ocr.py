from modules.ocr_analysis import extract_text_from_image

image_path = input("Enter the image path: ").strip().strip('"').strip("'")

with open(image_path, "rb") as image_file:
    extracted_text = extract_text_from_image(image_file)

print("\nExtracted Customer Feedback:")
print("--------------------------------")
print(extracted_text)