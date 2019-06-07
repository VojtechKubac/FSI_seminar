h=0.08;
myhext=h;
//myhprecis=0.003;
myhprecis=myhext/5.;

cx = 0.2;
cy = 0.2;
L_struct = 0.35101;
W_struct = 0.02;
r = 0.05;

Point(1) = {cx, cy, 0., myhprecis};
Point(2) = {0.6-L_struct, 0.19, 0., myhprecis};
Point(3) = {0.6-L_struct, 0.19+W_struct, 0., myhprecis};
Point(4) = {cx-r, cy, 0., myhprecis};

Point(5) = {0.6, 0.19, 0., myhprecis};
Point(6) = {0.6, 0.2, 0., myhprecis};
Point(7) = {0.6, 0.19+W_struct, 0., myhprecis};

Line(1) = {2, 5};
Line(2) = {5, 6};
Line(3) = {6, 7};
Line(4) = {7, 3};

// surface structure
Circle(5) = {3, 1, 2};
Line Loop(10) = {1, 2, 3, 4, 5};
Plane Surface(1) = {10};

Physical Line("1") = {5};            // Dirichlet
Physical Line("2") = {1, 2, 3, 4};   // rest

Physical Surface(1) = {1};         // solid

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

