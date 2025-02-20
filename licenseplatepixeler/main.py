import sys

def main():
    if len(sys.argv) > 1:
        from cli import run_cli
        run_cli()
    else:
        from gui import run_gui
        run_gui()

if __name__ == "__main__":
    main()
