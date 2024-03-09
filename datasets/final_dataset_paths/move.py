import os

# List of directories
dirs = [
    "dn-gan",
    "dn-sd14",
    "dn-glide",
    "dn-mj",
    "dn-dallemini",
    "dn-tt",
    "dn-sd21",
    "dn-cips",
    "dn-biggan",
    "dn-vqdiff",
    "dn-diffgan",
    "dn-sg3",
    "dn-gansformer",
    "dn-dalle2",
    "dn-ld",
    "dn-eg3d",
    "dn-projgan",
    "dn-sd1",
    "dn-ddg",
    "dn-ddpm",
    # Add more directories as needed
]

# Output file
output_file = "dn-joint-synth/val.txt"

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
        else:
            print(f"No train.txt found in {dir}")