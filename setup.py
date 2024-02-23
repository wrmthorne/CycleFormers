from setuptools import find_packages, setup


__version__ = '0.0.0.dev0'

REQUIRED_PKGS = [
    'transformers>=4.31.0',
    'torch>=1.4.0',
    'datasets'
]

EXTRAS = {
    'peft': ['peft>=0.4.0'],
}

setup(
    name='cycleformers',
    license='CC-BY-4.0 license',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Creative Commons Public License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
    ],
    url='UPDATE GITHUB URL',
    packages=find_packages(),
    include_package_data=True,
    install_requires=REQUIRED_PKGS,
    extras_require=EXTRAS,
    python_requires='>=3.10', # See what the lowest version of Python is that we can use
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    version=__version__,
    description='Train transformer language models with cycle consistency training.',
    keywords='transformers, pytorch, cycle consistency training, language modeling',
    author='William Thorne',
    author_email='wthorne1@sheffield.ac.uk',
)