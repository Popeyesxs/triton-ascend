add_triton_library(RemToMaskConversion
  RemToMaskConversionPass.cpp

  DEPENDS
  RemToMaskConversionPassIncGen

  LINK_LIBS
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRSupport
  TritonIR
)
