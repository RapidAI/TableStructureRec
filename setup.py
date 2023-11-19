# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import re
import sys
from typing import List

import setuptools


def extract_version(message: str) -> str:
    pattern = r"\d+\.(?:\d+\.)*\d+"
    matched_versions = re.findall(pattern, message)
    if matched_versions:
        return matched_versions[0]
    return ""


def read_txt(txt_path: str) -> List:
    if not isinstance(txt_path, str):
        txt_path = str(txt_path)

    with open(txt_path, "r", encoding="utf-8") as f:
        data = list(map(lambda x: x.rstrip("\n"), f))
    return data


MODULE_NAME = "lineless_table_rec"

if len(sys.argv) > 2:
    argv_str = "".join(sys.argv[2:])
    version = extract_version(argv_str)
else:
    version = "2."

sys.argv = sys.argv[:2]

setuptools.setup(
    name=MODULE_NAME,
    version=version,
    platforms="Any",
    description="无线表格还原库",
    author="SWHL",
    author_email="liekkaskono@163.com",
    install_requires=read_txt("requirements.txt"),
    include_package_data=True,
    packages=[MODULE_NAME, f"{MODULE_NAME}.models"],
    package_data={"": ["*.onnx"]},
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.6,<3.12",
    entry_points={
        "console_scripts": [f"{MODULE_NAME}={MODULE_NAME}.main:main"],
    },
)
