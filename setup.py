from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='Embed_adapter',
    version='0.1.1',
    author='Pavan Kumar Medarametla',
    author_email='pavankumarchowdary35@gmail.com',
    description='Python Package to Train Embedding adapter on top of any Embedding models from Hugging face and OpenAI',
    license='MIT',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)