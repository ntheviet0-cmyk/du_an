import io
from PIL import Image

# Read the generated image
img = Image.open(r"C:/Users/Public/graph_architecture.png")

# Write to destination using bytes path
dest = b"C:\\D\\xc4\\x91 \\xe1\\x80\\x81n c\\xc3\\xb4ng ngh\\xc4\\x81 th\\xc3\\xb4ng tin\\hr_causal_ai\\reports\\ibm\\graph_architecture.png"

import os
# Try to create the directory if needed
dir_part = b"C:\\D\\xc4\\x91 \\xe1\\x80\\x81n c\\xc3\\xb4ng ngh\\xc4\\x81 th\\xc3\\xb4ng tin\\hr_causal_ai\\reports\\ibm"
if not os.path.exists(dir_part):
    os.makedirs(dir_part)

with open(dest, "wb") as f:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    f.write(buf.getvalue())

print("Saved successfully")
