import os

# List of directories
dirs = [
    # "db-gan",
    "db-real",
    # "dn-gan",
    # "dn-sd14",
    # "dn-glide",
    # "dn-mj",
    # "dn-dallemini",
    # "dn-tt",
    # "dn-sd21",
    # "dn-cips",
    # "dn-biggan",
    # "dn-vqdiff",
    # "dn-diffgan",
    # "dn-sg3",
    # "dn-gansformer",
    # "dn-dalle2",
    # "dn-ld",
    # "dn-eg3d",
    # "dn-projgan",
    # "dn-sd1",
    # "dn-ddg",
    # "dn-ddpm",
    # Add more directories as needed
]
dirs.extend(["dn-real"] * 20)

real_count = 0
synth_count = 0
for dir in dirs:
    if "real" in dir:
        real_count += 1
    else:
        synth_count += 1
print(f"Real: {real_count}, Synthetic: {synth_count}")

# Output file
output_file = "oracle-real/val.txt"

# Open the output file in append mode
with open(output_file, 'a') as outfile:
    # Loop over directories
    for dir in dirs:
        # Construct the path to train.txt in the current directory
        train_txt = os.path.join(dir, "val.txt")
        # Check if train.txt exists in the directory
        if os.path.isfile(train_txt):
            # Open train.txt and append its lines to the output file
            with open(train_txt, 'r') as infile:
                outfile.write(infile.read())
            print(f"Appended {train_txt} to {output_file}")
        else:
            print(f"No train.txt found in {dir}")
            raise FileNotFoundError(f"No train.txt found in {dir}")