"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
"""
# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
import os
import re

ROOT = os.path.dirname(__file__)
VERSION_RE = re.compile(r'''__version__ = ['"]([0-9.]+)['"]''')

requires = [
    'numpy>=1.12.1',
    'tensorflow>=1.0.0',
    'nose',
]


# Define package version
def get_version():
    init = open(os.path.join(ROOT, 'facefinder', '__init__.py')).read()
    return VERSION_RE.search(init).group(1)


# Get the long description from the README file
def readme():
    open('README.rst').read()


setup(
    name='facefinder',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html

    version=get_version(),

    description='Face Detection using deep learning',

    long_description=readme(),

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',

        'Natural Language :: English',

        'Topic :: Software Development :: Libraries :: Python Modules',
        'Environment :: Plugins',
    ],

    keywords=['face detection', 'machine learning', 'deep learning', 'facefinder', 'akkefa'],

    url='https://github.com/akkefa/facefinder',

    author='Ikram Ali',

    author_email='mrikram1989@gmail.com',

    license='Apache License 2.0',

    packages=find_packages(),

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=requires,

    test_suite='nose.collector',

    tests_require=['nose', ],

    entry_points={
        'console_scripts': [
            'facefinder = facefinder.commands.dev:print_development'
        ],
    },

    include_package_data=True,

    zip_safe=False
)
