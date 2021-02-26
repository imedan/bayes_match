from setuptools import setup
import os


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="bayes_match",
    version="0.1.0",
    author="Ilija Medan",
    author_email="medan@astro.gsu.edu",
    description="Bayesian Cross-Match to Gaia DR2 Sources",
    long_description=read('README.md'),
    license="BSD 3-Clause",
    py_modules=['bayes_match.bayes_match']
    # classifiers=[]
)
