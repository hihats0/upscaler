"""python -m upscaler watch [secenekler]"""
import sys


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] != "watch":
        print("kullanim: python -m upscaler watch [--info] [--split] [--seconds N] ... (--help)")
        sys.exit(2)
    from .watch import main as watch_main
    watch_main(sys.argv[2:])


if __name__ == "__main__":
    main()
