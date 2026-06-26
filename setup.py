from setuptools import setup

setup(name='boostaroota',
      version='1.2.0.b',
      description='A Fast XGBoost Feature Selection Algorithm',
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