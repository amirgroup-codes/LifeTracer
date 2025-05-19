import os
import shutil
from setuptools import setup, find_packages, Command
import fnmatch
import sys


with open("README.md", "r") as f:
    long_description = f.read()

class CleanCommand(Command):
    """Custom clean command to tidy up the project root and remove build artifacts."""
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        # Directories and patterns to clean
        directories = ['./build', './dist', './.eggs']
        patterns = ['./*.egg-info', './*.pyc', './*.pyo', './__pycache__']

        for directory in directories:
            if os.path.exists(directory):
                shutil.rmtree(directory)
                print(f"Removed directory: {directory}")

        for root, dirs, files in os.walk('.'):
            for pattern in patterns:
                for filename in fnmatch.filter(files, pattern):
                    file_path = os.path.join(root, filename)
                    os.remove(file_path)
                    print(f"Removed file: {file_path}")

            # Optionally, remove __pycache__ directories recursively
            if '__pycache__' in dirs:
                shutil.rmtree(os.path.join(root, '__pycache__'))
                print(f"Removed __pycache__ in: {root}")

        print("Cleaned up the project directories and files.")

setup(
    name='lifetracer',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'pandas==2.2.3',
        'numpy==2.2.4',
        'loguru==0.7.3',
        'tqdm==4.67.1',
        'seaborn==0.13.2',
        'scipy==1.15.2',
        'scikit-learn==1.6.1',
        'plotly==6.0.1',
        'statsmodels==0.14.4',
        'xgboost==3.0.0',
        'svgpathtools==1.6.1',
    ],
    python_requires=">=3.10.8",
    long_description=long_description,
    long_description_content_type="text/markdown",
    cmdclass={
        'clean': CleanCommand,
    },
    entry_points={
        'console_scripts': [
            'lifetracer=lifetracer.__main__:main',
        ],
    },
    author="Daniel Saeedi",
    author_email="dsaeedi3@gatech.edu"
)