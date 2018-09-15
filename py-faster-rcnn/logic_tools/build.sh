cd ../

rm ./lib/nms/*.so
rm ./lib/pycocotools/*.so
rm ./lib/utils/*.so

cd ./lib

make

cd ..

rm ./lib/nms/cpu_nms.c
rm ./lib/nms/gpu_nms.cpp
rm ./lib/pycocotools/_mask.c
rm ./lib/utils/bbox.c

cd logic_tools
pwd




