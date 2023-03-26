from setuptools import setup

setup(
    name='pix_utils',
    version='0.6.3',
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        'pandas',
        'numpy',
        'scipy',
        'setuptools',
        'pytz'
    ]
)
