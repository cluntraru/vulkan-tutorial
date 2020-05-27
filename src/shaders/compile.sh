#!/bin/bash
glslc -fshader-stage=vertex vertex.glsl -o vert.spv
glslc -fshader-stage=fragment fragment.glsl -o frag.spv

