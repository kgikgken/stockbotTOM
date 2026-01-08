
from utils.screener import run_screening
from utils.report import generate_report

def main():
    result = run_screening()
    text = generate_report(result)
    print(text)

if __name__ == "__main__":
    main()
