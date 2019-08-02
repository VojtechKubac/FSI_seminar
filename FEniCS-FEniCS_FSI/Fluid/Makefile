#!/usr/bin/env gmake
SHELL=/bin/bash
.SECONDARY:

NAME:= mesh_Fluid

all: meshes

meshes: $(NAME)_L0  $(NAME)_L1  $(NAME)_L2    $(NAME)_L3 $(NAME)_L4

$(NAME)%: $(NAME)%.h5
	echo $< $@ $* $^

%.h5: %.xml 
	python3 mesh_process.py --mesh $*

%_L0.msh: %.geo
	gmsh -format msh2 -2 -setnumber levels 0 -o $@ $< 

%_L1.msh: %.geo
	gmsh -format msh2 -2 -setnumber levels 1 -o $@ $< 

%_L2.msh: %.geo
	gmsh -format msh2 -2 -setnumber levels 2 -o $@ $< 

%_L3.msh: %.geo
	gmsh -format msh2 -2 -setnumber levels 3 -o $@ $< 

%_L4.msh: %.geo
	gmsh -format msh2 -2 -setnumber levels 4 -o $@ $< 

%.xml : %.msh 
	dolfin-convert $< $@

#%.xml %_physical_region.xml %_facet_region.xml : %.msh
#	dolfin-convert $*.msh $*.xml

clean:
	rm *.msh
	rm *.xml
	rm *.xdmf
	rm *h5
	#rm mesh_ALE_L?.h5
#	rm -rf $(DATA)
