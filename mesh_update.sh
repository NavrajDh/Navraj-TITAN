#!/bin/bash 
set -eu 
cd /home/navraj/TITAN/simulation/PATO_0
gmshToFoam mesh/mesh.msh 
cp -rf constant/polyMesh constant/subMat1 
decomposePar -region subMat1
