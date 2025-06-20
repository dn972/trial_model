# library
set( LIB_NAME vvenc )

# create upper case name
string( TOUPPER ${LIB_NAME} LIB_NAME_UC )

# create file version.h
if(PROJECT_VERSION_RC)
  configure_file( version-rc.h.in "${CMAKE_BINARY_DIR}/${LIB_NAME}/version.h" )
else()
  configure_file( version.h.in "${CMAKE_BINARY_DIR}/${LIB_NAME}/version.h" )
endif()

# get source files
file( GLOB BASE_SRC_FILES "*.cpp"  "../CommonLib/*.cpp"  "../Utilities/*.cpp" "../DecoderLib/*.cpp" "../EncoderLib/*.cpp" )

# get include files
file( GLOB BASE_INC_FILES "*.h" "../CommonLib/*.h"  "../Utilities/*.h" "../DecoderLib/*.h" "../EncoderLib/*.h" "../apputils/*.h" )


if( NOT DEFINED VVENC_ENABLE_X86_SIMD )
  message( FATAL_ERROR "VVENC_ENABLE_X86_SIMD is not defined. Please include TopLevel CMakeLists.txt file of vvenc to set all dependencies!" )
endif()

if( VVENC_ENABLE_X86_SIMD )
  # get x86 source files
  file( GLOB X86_SRC_FILES "../CommonLib/x86/*.cpp" )

  # get x86 include files
  file( GLOB X86_INC_FILES "../CommonLib/x86/*.h" )

  ## get avx source files
  #file( GLOB AVX_SRC_FILES "../CommonLib/x86/avx/*.cpp" )

  # get avx2 source files
  file( GLOB AVX2_SRC_FILES "../CommonLib/x86/avx2/*.cpp" )

  # get sse4.1 source files
  file( GLOB SSE41_SRC_FILES "../CommonLib/x86/sse41/*.cpp" )

  ## get sse4.2 source files
  #file( GLOB SSE42_SRC_FILES "../CommonLib/x86/sse42/*.cpp" )
endif()

# get public/extern include files
file( GLOB PUBLIC_INC_FILES  "../../../include/${LIB_NAME}/*.h" )

# get all source files
set( SRC_FILES ${BASE_SRC_FILES} ${X86_SRC_FILES} )

# get all include files
file( GLOB PRIVATE_INC_FILES ${BASE_INC_FILES} ${X86_INC_FILES}  )

set( INC_FILES ${PRIVATE_INC_FILES} ${PUBLIC_INC_FILES}  )

# NATVIS files for Visual Studio
if( MSVC )
  file( GLOB NATVIS_FILES "../../VisualStudio/*.natvis" )

  # example: place header files in different folders
  source_group( "Natvis Files" FILES ${NATVIS_FILES} )
endif()

source_group( "Header Files"          FILES ${PUBLIC_INC_FILES} )
source_group( "Header Files\\private" FILES ${PRIVATE_INC_FILES} )

# set PRIVATE defines for all targets in this directory
add_compile_definitions( ${LIB_NAME_UC}_SOURCE )

# set PRIVATE include directories for all targets in this directory
include_directories( $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../include> $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}> )
include_directories( . .. ../DecoderLib ../EncoderLib ../CommonLib ../CommonLib/x86 ../apputils )
include_directories( SYSTEM ../../../thirdparty )

# set common warning flags
add_compile_options( "$<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:-Wall;-Werror;-Wno-deprecated-register;-Wno-unused-const-variable;-Wno-unknown-attributes>" )
add_compile_options( "$<$<CXX_COMPILER_ID:GNU>:-Wall;-Werror;-Wno-unused-function;;-Wno-unused-variable;;-Wno-sign-compare;;-fdiagnostics-show-option;-Wno-ignored-attributes>" )
add_compile_options( "$<$<CXX_COMPILER_ID:MSVC>:/W4;/WX;/wd4100;/wd4127;/wd4244;/wd4245;/wd4251;/wd4310;/wd4389;/wd4456;/wd4457;/wd4458;/wd4459;/wd4505;/wd4701;/wd4702;/wd4703;/wd4996>" )

# don't export all symbols from shared libraries by default (gcc: -fvisibility=hidden), only those marked as VVENC_DECL
#  behavior similar to __declspec(dllexport) on windows
set( CMAKE_C_VISIBILITY_PRESET hidden )
set( CMAKE_CXX_VISIBILITY_PRESET hidden )

set( CMAKE_VISIBILITY_INLINES_HIDDEN TRUE )

if( VVENC_ENABLE_X86_SIMD )
  # set needed compile definitions
  set_property( SOURCE ${SSE41_SRC_FILES} APPEND PROPERTY COMPILE_DEFINITIONS USE_SSE41 )
  #set_property( SOURCE ${SSE42_SRC_FILES} APPEND PROPERTY COMPILE_DEFINITIONS USE_SSE42 )
  #set_property( SOURCE ${AVX_SRC_FILES}   APPEND PROPERTY COMPILE_DEFINITIONS USE_AVX )
  set_property( SOURCE ${AVX2_SRC_FILES}  APPEND PROPERTY COMPILE_DEFINITIONS USE_AVX2 )
  # set needed compile flags
  if( MSVC )
    #set_property( SOURCE ${AVX_SRC_FILES}   APPEND PROPERTY COMPILE_FLAGS "/arch:AVX" )
    set_property( SOURCE ${AVX2_SRC_FILES}  APPEND PROPERTY COMPILE_FLAGS "/arch:AVX2" )
  elseif( UNIX OR MINGW )
    include( vvencCompilerSupport )

    set_if_compiler_supports_flag( FLAG_mxsave -mxsave  )
    set_if_compiler_supports_flag( FLAG_msse41 -msse4.1 )
    #set_if_compiler_supports_flag( FLAG_msse42 -msse4.2 )
    #set_if_compiler_supports_flag( FLAG_mavx   -mavx    )
    set_if_compiler_supports_flag( FLAG_mavx2  -mavx2   )

    set_property( SOURCE ${X86_SRC_FILES}   APPEND PROPERTY COMPILE_FLAGS ${FLAG_mxsave} )
    set_property( SOURCE ${SSE41_SRC_FILES} APPEND PROPERTY COMPILE_FLAGS "${FLAG_msse41}" )
    #set_property( SOURCE ${SSE42_SRC_FILES} APPEND PROPERTY COMPILE_FLAGS "${FLAG_msse42}" )
    #set_property( SOURCE ${AVX_SRC_FILES}   APPEND PROPERTY COMPILE_FLAGS "${FLAG_mavx}"   )
    set_property( SOURCE ${AVX2_SRC_FILES}  APPEND PROPERTY COMPILE_FLAGS "${FLAG_mavx2}"  )
  endif()

  #add_library( ${LIB_NAME}_x86_simd OBJECT ${SSE41_SRC_FILES} ${SSE42_SRC_FILES} ${AVX_SRC_FILES} ${AVX2_SRC_FILES} )
  add_library( ${LIB_NAME}_x86_simd OBJECT ${SSE41_SRC_FILES} ${AVX2_SRC_FILES} )
  # disble LTO for the files compiled with special architecture flags
  set_target_properties( ${LIB_NAME}_x86_simd PROPERTIES
                                              INTERPROCEDURAL_OPTIMIZATION                OFF
                                              INTERPROCEDURAL_OPTIMIZATION_RELEASE        OFF
                                              INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO OFF
                                              INTERPROCEDURAL_OPTIMIZATION_MINSIZEREL     OFF )

  set_target_properties( ${LIB_NAME}_x86_simd PROPERTIES FOLDER lib )
endif()

# set resource file for MSVC compilers
if( MSVC )
  set( RESOURCE_FILE ${LIB_NAME}.rc )
endif()

add_library( ${LIB_NAME} ${SRC_FILES} $<$<TARGET_EXISTS:${LIB_NAME}_x86_simd>:$<TARGET_OBJECTS:${LIB_NAME}_x86_simd>> ${INC_FILES} ${NATVIS_FILES} ${RESOURCE_FILE} )

target_compile_definitions( ${LIB_NAME} PUBLIC $<$<STREQUAL:$<TARGET_PROPERTY:${LIB_NAME},TYPE>,SHARED_LIBRARY>:${LIB_NAME_UC}_DYN_LINK> )

target_include_directories( ${LIB_NAME} SYSTEM INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../include> $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}> )

if( CMAKE_COMPILER_IS_GNUCC AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.0 )
  target_compile_options( ${LIB_NAME} PRIVATE $<$<CXX_COMPILER_ID:GNU>: -Wno-array-bounds> )
endif()


if( VVENC_ENABLE_THIRDPARTY_JSON )
  set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DVVENC_ENABLE_THIRDPARTY_JSON" )
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVVENC_ENABLE_THIRDPARTY_JSON" )
  target_include_directories( ${LIB_NAME} PRIVATE ../../../thirdparty/nlohmann_json/single_include )
endif()

if( TARGET INTEL_ITT)
  target_link_libraries( ${LIB_NAME} PRIVATE INTEL_ITT Threads::Threads )
else()
  target_link_libraries( ${LIB_NAME} PRIVATE Threads::Threads )
endif()

# set the folder where to place the projects
set_target_properties( ${LIB_NAME} PROPERTIES
                                   VERSION ${PROJECT_VERSION}
                                   SOVERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
                                   FOLDER lib )
