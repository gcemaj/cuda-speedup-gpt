import typer

from collections import defaultdict
from typing import Optional
from nyu.gpu.speedup.generator.driver.driver import Driver


def main():
    driver = Driver()
    driver.generate_data(1)

if __name__ == "__main__":
    typer.run(main)