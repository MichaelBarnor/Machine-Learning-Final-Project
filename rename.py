import os

folder_path = 'temp' 

for filename in os.listdir(folder_path):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        name, ext = os.path.splitext(filename)
        new_name = f"{name}(1){ext}"
        src = os.path.join(folder_path, filename)
        dst = os.path.join(folder_path, new_name)
        os.rename(src, dst)

print("Renaming complete!")
