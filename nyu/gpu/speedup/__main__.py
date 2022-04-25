import typer

from collections import defaultdict
from typing import Optional
from nyu.gpu.speedup.generator.driver.driver import Driver

def main(datapoints : int):
    driver = Driver()
    driver.generate_data(datapoints)

if __name__ == "__main__":
    typer.run(main)
