import os

from setuptools import setup

PATH_ROOT = os.path.dirname(__file__)


def load_requirements(
    path_dir=PATH_ROOT, file_name="requirements.txt", comment_char="#"
):
    with open(os.path.join(path_dir, file_name), "r") as file:
        lines = [ln.strip() for ln in file.readlines()]
    reqs = []
    for ln in lines:
        if comment_char in ln:  # filer all comments
            ln = ln[: ln.index(comment_char)].strip()
        if ln:  # if requirement is not empty
            reqs.append(ln)

    return reqs


reqs = load_requirements()

setup(
    name="cliv",
    version="0.1.0",
    description="Continual learning of inter-reader variability for medical image segmentation (CLIV) project",
    author="Matthias Perkonigg",
    packages=["cliv"],
    include_package_data=True,
    zip_safe=False,
    python_requires=">=3.9",
    setup_requires=[],
    install_requires=reqs,
)
