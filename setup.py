# setup.py
from setuptools import setup, find_packages

setup(
    name="llam",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "fastapi",
        "uvicorn",
        "pydantic",
        "langchain",
        "python-dotenv",
        # Add other dependencies as needed
    ],
    entry_points={
        'console_scripts': [
            'LLAM-start=LLAM.server:main',
        ],
    },
)
