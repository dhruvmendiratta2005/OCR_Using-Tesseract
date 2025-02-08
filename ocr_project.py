import cv2
import pytesseract
import numpy as np
import matplotlib.pyplot as plt

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = R"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load the image
image_path = R"D:\OCR\Picture_003.jpg"  # Use raw string to avoid path errors
img = cv2.imread(image_path, 0)  # Load in grayscale
 # Enhance contrast
# Check if image is loaded
if img is None:
    raise FileNotFoundError("Image not found. Check the file path.")

# Display the image
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.show()

# Sharpen the image
def sharpen_image(im):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # Sharpening kernel
    im = cv2.filter2D(im, -1, kernel)
    return im

img = sharpen_image(img)

plt.imshow(img, cmap='gray')
plt.title("Sharpened Image")
plt.show()

# Apply adaptive thresholding
img_thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# Invert for better OCR detection
img_thresh = 255 - img_thresh

plt.imshow(img_thresh, cmap='gray')
plt.title("Thresholded Image")
plt.show()

# Align text
def align_text(im):
    coords = np.column_stack(np.where(im > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    h, w = im.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(im, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

img = align_text(img_thresh)

plt.imshow(img, cmap='gray')
plt.title("Aligned Image")
plt.show()

# Split Image into Rows
a = np.sum(img == 255, axis=1)
rows = []
seg = []

for i in range(len(a)):
    if a[i] > 0:
        seg.append(i)
    if (a[i] == 0) and (len(seg) >= 5):
        rows.append(seg)
        seg = []
if len(seg) > 0:
    rows.append(seg)

# Check if any text is detected
if not rows:
    print("No text detected.")
    exit()

# Display one detected line
plt.imshow(img[rows[0][0]:rows[0][-1], :], cmap='gray')
plt.title("One Line of Text")
plt.show()

# Extract Text Using OCR
custom_config = r'--oem 3 --psm 6'  # OCR Engine Mode and Page Segmentation Mode
output_text = []

for i, row in enumerate(rows):
    text_line = pytesseract.image_to_string(img[row[0]:row[-1], :], config=custom_config)
    text_line = text_line.strip()
    if text_line:
        output_text.append(text_line)
        print(f"Line {i+1}: {text_line}")

# Save extracted text to a file
if output_text:
    with open("extracted_text.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(output_text))
    print("Extracted text saved to extracted_text.txt")
else:
    print("No text extracted.")
