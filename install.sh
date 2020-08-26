pip install pyyaml
pip install numpy 
pip install scipy
pip install matplotlib
pip install cython
pip install opencv-python
pip install pycocotools
pip install pytest
pip install pybind11

cd layers/anchor_gen_cuda
python3 setup.py install
cd ../nms_cuda
python3 setup.py install
cd ../sigmoid_focal_loss_cuda
python3 setup.py install

cd ../../dataset/dbext
python3 setup.py install
