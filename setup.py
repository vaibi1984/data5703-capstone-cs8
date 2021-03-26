from setuptools import setup

setup(
    name="cs8-skin-cancer",
    version="0.0.1",
    author="CS8 Team",
    author_email="cehorn@stanford.edu",
    packages=["glrm"],
    package_dir={"core": "core"},
    url="http://github.com/cehorn/GLRM/",
    license="MIT",
    install_requires=["tensorflow >= 2.0",
                      "matplotlib"]
)
