from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="cs8-skin-cancer",
    version="0.0.1",
    author="CS8 Team",
    packages=["cs8-skin-cancer"],
    package_dir={"core": "core"},
    url="https://github.com/kirubhakaran12/data5703-capstone-cs8/",
    license="MIT",
    install_requires=requirements
)
