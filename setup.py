import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="EPViz",
    version="0.0.0",
    description="An open source EEG Visualization software",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jcraley/epviz",
    project_urls={
        "Bug Tracker": "https://github.com/jcraley/epviz/issues",
        "Documentation": "https://engineering.jhu.edu/nsa/links/epviz/",
    },
    package_data={'epviz.ui_files': ['gui_stylesheet.css',]},
    packages=["epviz", "epviz.signal_loading", "epviz.edf_saving",
        "epviz.filtering", "epviz.image_saving", "epviz.models",
        "epviz.predictions", "epviz.preprocessing", "epviz.signal_stats",
        "epviz.spectrogram_window","epviz.ui_files",],
    entry_points={
        'console_scripts': ['epviz=epviz.plot:main', ]
    },
)
