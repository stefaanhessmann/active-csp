import os
import io

from setuptools import setup, find_packages


def read(fname):
    with io.open(os.path.join(os.path.dirname(__file__), fname), encoding="utf-8") as f:
        return f.read()


setup(
    name="activecsp",
    version="0.1",
    author="Stefaan Hessmann",
    packages=find_packages("src"),
    package_dir={"": "src"},
    scripts=[
        "src/scripts/main.py",
        "src/scripts/compute_reference_labels.py",
        "src/scripts/compute_representations.py",
        "src/scripts/pool_optimization.py",
        "src/scripts/reference_optimization.py",
        "src/scripts/compute_random_labels.py",
        "src/scripts/stopping_criterion.py",
    ],
    install_requires=[],
    include_package_data=True,
    license="MIT",
    description="Active Learning for Crystal St",
    long_description="todo",
)
