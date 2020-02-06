from setuptools import setup
import os
import sys

_here = os.path.abspath(os.path.dirname(__file__))

if sys.version_info[0] < 3:
    with open(os.path.join(_here, 'README.md')) as f:
        long_description = f.read()
else:
    with open(os.path.join(_here, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()

version = {}
with open(os.path.join(_here, 'iblapps', 'version.py')) as f:
    exec(f.read(), version)

setup(
    name='iblapps',
    version=version['__version__'],
    description=('PyQt5 visualization apps for the IBL pipeline.'),
    long_description=long_description,
    author='IBL',
    author_email='info@internationalbrainlab.org',
    url='https://github.com/int-brain-lab/iblapps',
    license='MIT',
    packages=['iblapps'],
#   no dependencies in this example
#   install_requires=[
#       'dependency==1.2.3',
#   ],
#   no scripts in this example
#   scripts=['bin/a-script'],
    include_package_data=True,
    classifiers=[
        'Development Status :: 0 - Alpha',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.6+'
    ],
    )