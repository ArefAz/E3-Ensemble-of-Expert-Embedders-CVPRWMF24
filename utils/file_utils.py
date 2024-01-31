import os 

def list_files(directory, extensions, max_files):
    ext_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            final_path = os.path.join(root, file)
            for ext in extensions:
                if file.endswith(ext):
                    ext_files.append(final_path)
        if max_files is not None and len(ext_files) > max_files:
            ext_files = ext_files[:max_files]
            break
    return ext_files