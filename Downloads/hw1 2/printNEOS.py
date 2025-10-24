with open("sent.train", "r") as infile, open("NEOS", "w") as outfile:
    for line in infile:
        if line.startswith("NEOS"):
            outfile.write(line)
