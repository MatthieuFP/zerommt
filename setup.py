from pathlib import Path
from setuptools import find_packages, setup

if __name__ == "__main__":
    with Path(Path(__file__).parent, "README.md").open(encoding="utf-8") as file:
        long_description = file.read()

    setup(
        name='zerommt',
        version='0.1.0',
        description='An open-source framework for zero-shot multimodal machine translation inference',
        long_description=long_description,
        long_description_content_type="text/markdown",
        data_files=[(".", ["README.md"])],
        author='Matthieu Futeral',
        license='MIT',
        packages=find_packages(),
        install_requires=['numpy==1.24.2',
                          'torch==2.1.1',
                          'torchvision==0.16.1',
                          'transformers==4.35.2',
                          'open-clip-torch==2.24.0',
                          'timm==0.9.16',
                          'huggingface-hub==0.21.2',
                          'accelerate==0.26.1',                 
                        ],
        dependency_links=[
            'https://download.pytorch.org/whl/cu121'
        ],
        classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'License :: OSI Approved :: MIT License',      
            'Programming Language :: Python :: 3.9',
        ],
    )