from PIL import Image

images ["image1.png", "image2.png", "image3.png", "image4.png"]

image_list = []

for file in images:
    img Image.open(file).convert("RGB")
    image list.append(img)

image_list[0].save(
    "output.pdf",
    save_all=True,
    append_images=image_list[1:]
)
print("PDF created successfully!")
