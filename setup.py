from setuptools import setup, find_packages

with open('requirements.txt') as f:
    require = [x.strip() for x in f.readlines() if not x.startswith('git+')]

setup(
    name='iblapps',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=require,
    entry_points={
        'console_scripts': [
            'atlas=atlasview.atlasview:main',
            'align=atlaselectrophysiology.ephys_atlas_gui:main',
        ]
    },
)
