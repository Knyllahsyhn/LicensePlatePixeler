import sys
import argparse
from src.cli import run_cli
from src.gui import run_gui

def main():
    parser = argparse.ArgumentParser(description="License Plate Pixelation Tool")
    parser.add_argument("-i", "--input", type=str, help="Path to input video or folder")
    parser.add_argument("-o", "--output", type=str, help="Path to output folder")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.input and args.output:
        run_cli(args.input, args.output, args.verbose)
    else:
        run_gui()

if __name__ == "__main__":
    main()
