set -e
export BASEDIR="$(cd "$(dirname ${BASH_SOURCE[0]})" ;  pwd -P)"
cd "${BASEDIR}"

# Create and source virtualenv
if [ -e "${BASEDIR}/venv/bin/activate" ]; then
	echo "using existing virtualenv"
else	
	echo "creating virtualenv ..."
	virtualenv --python=python3 venv
fi

source venv/bin/activate

# Upgrade pip and install libraries (dependencies are also installed)
pip install --upgrade pip

pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu
pip install git+https://gitlab.mathematik.uni-stuttgart.de/pub/ians-anm/pde-vkoga@v0.1.1
pip install git+https://github.com/GabrieleSantin/VKOGA.git
pip3 install scipy six matplotlib
pip3 install pytorch-lightning
pip3 install pandas
# pip3 install tensorflow
pip3 install scikit-learn


