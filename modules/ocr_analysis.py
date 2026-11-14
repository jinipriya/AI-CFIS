
import pytesseract
import cv2
import numpy as np
from PIL import Image

# Configure Tesseract installation path
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


def extract_text_from_image(image_file):
    """
    Extract customer feedback text from an uploaded image.
    """

    image = Image.open(image_file).convert("RGB")

    # Convert PIL image to OpenCV format
    image_array = np.array(image)
    image_cv = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    # Convert to grayscale
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to improve text recognition
    processed_image = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    # Extract text using Tesseract
    extracted_text = pytesseract.image_to_string(
        processed_image,
        config="--oem 3 --psm 6"
    )

    return extracted_text.strip()
