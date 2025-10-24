import argparse

def extract_nnhead(input_file, output_file):
    """Extracts NNHEAD lines and writes them to an output file."""
    with open(input_file, "r") as fin, open(output_file, "w") as fout:
        for line in fin:
            arr = line.strip().split('\t', 1)
            if arr[0] == 'PTEXT':  # Only save NNHEAD lines
                fout.write(arr[1] + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to the input data file")
    parser.add_argument("--output", default="nnhead_output.txt", help="Path to the output file")
    args = parser.parse_args()

    extract_nnhead(args.input, args.output)
    print(f"ptext lines saved to {args.output}")
