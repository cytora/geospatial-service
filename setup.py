from distutils.core import setup

from setuptools import find_packages

setup(
    name='geospatial-service',
    version='2021.06.24',
    description='Geospatial Service',
    author='cytora',
    entry_points={
        'console_scripts': [
            'run-service = service.handler:run',
        ]
    },
    packages=find_packages(
        include=[
            'configs',
            'configs.*',
            'service',
            'service.*',
            'models',
            'models.*',
            'generated',
            'generated.*'
        ],
        exclude=[
            'tests',
        ],
    ),
    package_data={
        'configs': [
            '*'
        ],
    },
    setup_requires=[
        'pytest-runner',
    ],
    install_requires=[
        'datadog==0.41.0',
        'py-platform-utils[app-sanic]==0.0.24',
        'pydantic==1.8.1',
    ],
    dependency_links=[
        'git+ssh://git@github.com/cytora/py-platform-utils@0.0.24#egg=py-platform-utils-0.0.24',
    ],
    extras_require={
        'dev': [
            'pylint==2.3.1',
            'pytest==3.6.4',
            'pytest-mock==1.10.0',
            'pytest-asyncio==0.9.0',
        ],
    },
)
