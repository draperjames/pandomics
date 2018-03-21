# -*- coding: utf-8 -*-
import os
import re
# import glob
import shutil
import subprocess


here = os.getcwd()


def native_cmd(cmd, whitespace=False):
    """Returns the output of a native command on a give system.

    NOTE: Results may contain white space charaters at end of lines.
    """
    result = subprocess.check_output(cmd, shell=True).decode()
    if whitespace:
        return result
    else:
        # Remove carrige returns.
        result = re.sub('\r|\n', '', result, re.X)
        return result


def main():
    if os.path.exists("docs"):
        try:
            shutil.move("docs", "html")
        except FileNotFoundError as err:
            print(err)

    if os.path.exists("html"):
        try:
            native_cmd("html\make.bat html")

        except Exception as err:
            print(err)

        try:
            shutil.move("html", "docs")
        except FileNotFoundError as err:
            print(err)


if __name__ == "__main__":
    main()
