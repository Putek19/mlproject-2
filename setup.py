from setuptools import find_packages,setup
from typing import List


def get_requirements():
    """
    This function will return list of requirements
    """
    requirement_lst = []
    try:
        with open("requirements.txt",'r') as file:
            lines = file.readlines()
            for line in lines:
                requirement = line.strip()
                if requirement and requirement != '-e .':
                    requirement_lst.append(requirement)
    except FileNotFoundError:
        print("Requirements.txt file not found")
    
    return requirement_lst

setup(
    name="Ml project",
    version='0.0.1',
    author="Jakub Nowacki",
    author_email='kubanowacki.jn@gmail.com',
    packages=find_packages(),
    install_requires = get_requirements()
)




