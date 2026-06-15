## Create virtual python environment for xspatula

To create a virtual python environment for xspatula you can use the file _xspatula_py_3.12.yml_.
Open a Terminal window and navigate to the folder that contains _xspatula_py_3.12.yml_. Then execute the command _conda env create_ as shown below. When you start a new notebook in the xspatula package, select the _xspatula_py_3.12.yml_ as the python environment.

## Create virtual python enviornment
conda env create --file xspatula_ai4sh_py_3.12.yml

## Remove virtual python environment
conda remove --name xspatula_ai4sh_py_3.12 --all

## Installing Cubist regressor

If you want to use the Cubist regressor for Machine Learning modeling, you have to install that manually after creting your virtual environment. In your terminal execture the following two commands to install Cubist:

conda activate xspatula_ai4sh_py_3.12

pip install --upgrade cubist