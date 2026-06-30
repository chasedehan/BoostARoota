from setuptools import setup
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(name='boostaroota',
      version='2.0.0',
      description='A Fast XGBoost Feature Selection Algorithm',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='http://github.com/chasedehan/BoostARoota',
      author='Chase DeHan',
      author_email='chasedehan@yahoo.com',
      license='MIT',
      packages=['boostaroota'],
      zip_safe=False,
      install_requires=[
          'numpy>=1.21,<3.0',
          'pandas>=1.5,<3.0',
          'xgboost>=1.7,<3.0',
          'scikit-learn>=1.3,<2.0',
      ])