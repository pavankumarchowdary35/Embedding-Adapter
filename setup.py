from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = f.read().splitlines()
setup(
    name='embedding_ada',
    version='0.1.9',
    author='Pavan Kumar Medarametla',
    author_email='pavankumarchowdary35@gmail.com',
    description='Python Package to Train Embedding adapter on top of any Embedding models from Hugging face and OpenAI',
    license='MIT',
    long_description=long_description,
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