import subprocess
import sys

def install():
    with open("pip.txt", "r") as file:
        for line in file:
            package = line.strip()

            if package == "":
                continue
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])


