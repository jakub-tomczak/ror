from setuptools import setup, find_packages
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding = 'utf-8') as f:
    long_description = f.read()

setup(
  name = 'ror',
  packages = ['ror'],
  version = '0.0.1',
  python_requires = '>=3.6',
  license = 'MIT',
  description = 'ROR method solver',
  long_description = long_description,
  long_description_content_type = 'text/markdown',
  author = 'Jakub Tomczak',
  url = 'https://github.com/jtomczak/ror',
  install_requires = [
    'gurobipy',
    'pandas',
    'numpy',
    'graphviz'
  ],
  dependency_links = ['https://pypi.gurobi.com'],
  classifiers = [
    'Development Status :: 3 - Alpha',
    'Programming Language :: Python :: 3',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent'
  ],
)