//SetFactory("OpenCASCADE");

h=0.07;
myhext=h;
//myhprecis=0.003;
myhprecis=myhext/5.;

L = 2.5;
W = 0.41;
cx = 0.2;
cy = 0.2;
L_struct = 0.35101;
W_struct = 0.02;
r = 0.05;

Point(1) = {0., 0., 0., myhext};
Point(2) = {L, 0., 0., myhext};
Point(3) = {L, W, 0., myhext};
Point(4) = {0., W, 0., myhext};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Point(5) = {cx, cy, 0., myhprecis};
Point(6) = {0.6-L_struct, 0.19, 0., myhprecis};
Point(7) = {0.6-L_struct, 0.19+W_struct, 0., myhprecis};
Point(8) = {cx-r, cy, 0., myhprecis};
Circle(5) = {7, 5, 8};
Circle(6) = {8, 5, 6};

Point(9) = {0.6, 0.19, 0., myhprecis};
Point(10) = {0.6, 0.19+W_struct, 0., myhprecis};

Point(11) = {0.6, 0.19+W_struct/2., 0, myhprecis};
Point(12) = {1.5, 0.19+W_struct/2., 0, myhprecis*3.};

// FSI interface
Line(7) = {6, 9};
Line(8) = {9, 11};
Line(9) = {11, 10};
Line(10) = {10, 7};

// symmetry line
Line(11) = {11, 12};


// surface fluid
Line Loop(11) = {3, 4, 1, 2};
Line Loop(12) = {5, 6, 7, 8, 9, 10, 11, -11};
Plane Surface(1) = {11, 12};

// surface structure
// Circle(14) = {7, 5, 6};
// Line Loop(15) = {10, 14, 7, 8, 9};
// Plane Surface(2) = {15};

//Physical Line("88") = {11};

Physical Line("1") = {4};            // inflow
Physical Line("2") = {1,3};          // rigid walls
Physical Line("3") = {5, 6};         // rigid part of circle
Physical Line("4") = {7, 8, 9, 10};  // FSI interface
Physical Line("5") = {2};            // outflow

//Physical Line("5") = {7,8,9,10};     // FSI interface
//Physical Line("6") = {5,6};          // fluid-cylinder interface
    // this would yield to overwriting of FEniCS MeshFunction (probably)

Physical Surface(0) = {1};         // fluid
// Physical Surface(1) = {2};         // solid

Mesh.Algorithm = 8; // Frontal
Mesh.Optimize=1;
Mesh.OptimizeThreshold=0.5;
Mesh.AnisoMax=10.0;
Mesh.Smoothing=100;
Mesh.OptimizeNetgen=1;

Mesh 2;

If ( levels > 0 )
For i In { 1 : levels }
  RefineMesh;
EndFor 
EndIf

