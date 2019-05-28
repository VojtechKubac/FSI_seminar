#!/usr/bin/env gmake
SHELL=/bin/bash
.SECONDARY:

DIR:= meshes
NAME:= mesh_structure

all: $(NAME)_L0 $(NAME)_L1 $(NAME)_L2  # $(NAME)2D_ns_L3 $(NAME)2D_ns_L4

$(NAME)%: $(DIR)/$(NAME)%.h5
	echo $< $@ $* $^

$(DIR):
	mkdir -p $(DIR)

$(DIR)/%.h5: $(DIR)/%.xml 
	python3 xml2h5.py --mesh $(DIR)/$*

$(DIR)/%_L0.msh: %.geo | $(DIR)
	gmsh -format msh2 -2 -setnumber levels 0 -o $@ $< 

$(DIR)/%_L1.msh: %.geo | $(DIR)
	gmsh -format msh2 -2 -setnumber levels 1 -o $@ $< 

$(DIR)/%_L2.msh: %.geo | $(DIR)
	gmsh -format msh2 -2 -setnumber levels 2 -o $@ $< 

$(DIR)/%_L3.msh: %.geo | $(DIR)
	gmsh -format msh2 -2 -setnumber levels 3 -o $@ $< 

$(DIR)/%_L4.msh: %.geo | $(DIR)
	gmsh -format msh2 -2 -setnumber levels 4 -o $@ $< 

$(DIR)/%.xml : $(DIR)/%.msh 
	dolfin-convert $< $@

#%.xml %_physical_region.xml %_facet_region.xml : %.msh
#	dolfin-convert $*.msh $*.xml

clean:
	rm -rf $(DIR)
	#rm *.msh
	#rm *.xml
	#rm *.xdmf
	#rm mesh_ALE_L?.h5
#	rm -rf $(DATA)

