// Gmsh project created on Mon Sep 18 15:21:03 2023
SetFactory("OpenCASCADE");
//+
Circle(1) = {0, 0, 0, 1, 0, 2*Pi};
//+
Circle(2) = {0, 0, 0, 0.8, 0, 2*Pi};
//+
Curve Loop(1) = {1};
//+
Curve Loop(2) = {2};
//+
Plane Surface(1) = {1, 2};
//+
Extrude {0, 0, 2} {
  Surface{1}; 
}
