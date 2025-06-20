# Top level CMakeLists.txt
#
# minimum required cmake version
cmake_minimum_required( VERSION 3.13.0 FATAL_ERROR )

# set cmake policy
if( NOT CMAKE_VERSION VERSION_LESS 3.13.0 )
  # Use latest policy
  cmake_policy( VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} )
endif()

# project name
project( vvenc VERSION 1.10.0 )

# set alternative version numbering for release candidates
#set( PROJECT_VERSION_RC rc1 )
if( PROJECT_VERSION_RC )
    set( PROJECT_VERSION "${PROJECT_VERSION}-${PROJECT_VERSION_RC}" )
endif()

set( CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/modules" )
message( STATUS "CMAKE_MODULE_PATH: updating module path to: ${CMAKE_MODULE_PATH}" )

# we enable x86 intrinsics for all target architectures, because they are implemented through simd-everywhere on non-x86
set( VVENC_ENABLE_X86_SIMD TRUE CACHE BOOL "enable x86 intrinsics" )

include( vvencCompilerSupport )

# enable sse4.1 build for all source files for gcc and clang
if( VVENC_ENABLE_X86_SIMD )
  if( UNIX OR MINGW )
    # when building for non-x86, but emulating simd using simd-everywhere (e.g. on ARM),
    # the x86-compiler flags are not understood by the compiler
    set_if_compiler_supports_flag( FLAG_msse41 -msse4.1 )
    add_compile_options( ${FLAG_msse41} )
  endif()

  if( NOT ${CMAKE_SYSTEM_NAME} STREQUAL "Emscripten" )
    check_missing_intrinsics()
  endif()

  set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DTARGET_SIMD_X86" )
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTARGET_SIMD_X86" )
endif()

if( NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR )
  # set exception handling
  if( MSVC )
    add_compile_options( "/EHsc" )
  endif()

  list( FIND ${PROJECT_NAME}_ADD_SUBDIRECTORIES "source/App/vvencFFapp" vvEncFfAppFound )
  list( FIND ${PROJECT_NAME}_ADD_SUBDIRECTORIES "source/App/vvencapp" vvEncAppFound )

  # vvenc embedded by superproject, always include source/Lib/vvenc  as first component
  list( PREPEND ${PROJECT_NAME}_ADD_SUBDIRECTORIES "source/Lib/vvenc" )
  list( REMOVE_DUPLICATES ${PROJECT_NAME}_ADD_SUBDIRECTORIES )
  message( STATUS "${CMAKE_CURRENT_SOURCE_DIR}: ${PROJECT_NAME} embedded, subdirectories to be added: ${${PROJECT_NAME}_ADD_SUBDIRECTORIES}" )
  # add subdirectories the superproject asked for
  foreach( subdir IN LISTS ${PROJECT_NAME}_ADD_SUBDIRECTORIES )
    add_subdirectory( ${subdir} )
  endforeach()
  return()
endif()

# enable install target
set( VVENC_ENABLE_INSTALL                   ON  CACHE BOOL   "Enable or disable install target" )

# enable postfix                                             
set( VVENC_ENABLE_BUILD_TYPE_POSTFIX        OFF CACHE BOOL   "Enable or disable build type postfix for apps and libs" )

set( VVENC_ENABLE_LINK_TIME_OPT             ON  CACHE BOOL   "Enable link time optimization for release and profile builds" )

set( VVENC_ENABLE_THIRDPARTY_JSON           ON  CACHE BOOL   "Enable use of thirdparty json library" )

set( VVENC_INSTALL_FULLFEATURE_APP		    OFF CACHE BOOL   "Install the full-feature app: vvencFFapp" )

if( CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR
    CMAKE_CXX_COMPILER_ID STREQUAL "Clang" )

  # enable address sanitizer
  set( VVENC_USE_ADDRESS_SANITIZER          OFF CACHE BOOL   "Enable or disable address sanitizer" )

  set( VVENC_OPT_TARGET_ARCH                ""  CACHE STRING "Enable or disable building with architecture specific optimization" )
endif()


if( VVENC_ENABLE_BUILD_TYPE_POSTFIX )
  if( BUILD_SHARED_LIBS )
    # set postfixes for shared libraries
    set( CMAKE_RELEASE_POSTFIX        "-s"   CACHE STRING "Set release library postfix" )
    set( CMAKE_DEBUG_POSTFIX          "-ds"  CACHE STRING "Set debug library postfix" )
    set( CMAKE_RELWITHDEBINFO_POSTFIX "-rds" CACHE STRING "Set relwithdebinfo library postfix" )
    set( CMAKE_MINSIZEREL_POSTFIX     "-mrs" CACHE STRING "Set minsizerel library postfix" )
  else()
    # set postfixes for static libraries
    set( CMAKE_RELEASE_POSTFIX        ""     CACHE STRING "Set release library postfix" )
    set( CMAKE_DEBUG_POSTFIX          "-d"   CACHE STRING "Set debug library postfix" )
    set( CMAKE_RELWITHDEBINFO_POSTFIX "-rd"  CACHE STRING "Set relwithdebinfo library postfix" )
    set( CMAKE_MINSIZEREL_POSTFIX     "-mr"  CACHE STRING "Set minsizerel library postfix" )
  endif()
endif()

# set VVENC_OUTPUT_DIR_POSTFIX
if( BUILD_SHARED_LIBS )
  set( VVENC_OUTPUT_DIR_POSTFIX shared )
else()
  set( VVENC_OUTPUT_DIR_POSTFIX static )
endif()

set( VVENC_ENABLE_TRACING         OFF CACHE BOOL "Set ENABLE_TRACING=1 as a compiler flag" )

set( VVENC_TOPLEVEL_OUTPUT_DIRS   ON  CACHE BOOL   "Put build artifacts into ${CMAKE_SOURCE_DIR}/{bin,lib}/" )

if( VVENC_TOPLEVEL_OUTPUT_DIRS )
  # Using CMake's default library name convention which is the same for all configurations.
  set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG          "${CMAKE_SOURCE_DIR}/lib/debug-${VVENC_OUTPUT_DIR_POSTFIX}" )
  set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE        "${CMAKE_SOURCE_DIR}/lib/release-${VVENC_OUTPUT_DIR_POSTFIX}" )
  set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_SOURCE_DIR}/lib/relwithdebinfo-${VVENC_OUTPUT_DIR_POSTFIX}" )
  set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY_MINSIZEREL     "${CMAKE_SOURCE_DIR}/lib/minsizerel-${VVENC_OUTPUT_DIR_POSTFIX}" )

  set( CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG          "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG}" )
  set( CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE        "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE}" )
  set( CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO}" )
  set( CMAKE_LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL     "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY_MINSIZEREL}" )

  set( CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG          "${CMAKE_SOURCE_DIR}/bin/debug-${VVENC_OUTPUT_DIR_POSTFIX}" )
  set( CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE        "${CMAKE_SOURCE_DIR}/bin/release-${VVENC_OUTPUT_DIR_POSTFIX}" )
  set( CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_SOURCE_DIR}/bin/relwithdebinfo-${VVENC_OUTPUT_DIR_POSTFIX}" )
  set( CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL     "${CMAKE_SOURCE_DIR}/bin/minsizerel-${VVENC_OUTPUT_DIR_POSTFIX}" )
else()
  # Put build artifacts below build-directory. (Don't need to add {debug,release,relwithdebinfo,minsizerel}-{static,shared}/)
  set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib" )
  set( CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib" )
  set( CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin" )
endif()

# enable or disable Intel Vtune ITT Tracing
if( CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo" )
    set( VVCEncoderLib_ENABLE_ITT ON CACHE BOOL "Enable Intel Runtime Support for Profiling" )
else()
    set( VVCEncoderLib_ENABLE_ITT OFF CACHE BOOL "Enable Intel Runtime Support for Profiling" )
endif()

# set default CMAKE_BUILD_TYPE to Release if not set
if( NOT CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE )
endif()

# set c++14
set( CMAKE_CXX_STANDARD 14 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )

# compile everything position independent (even static libraries)
set( CMAKE_POSITION_INDEPENDENT_CODE TRUE )

# set verbose compile options
#set( CMAKE_VERBOSE_MAKEFILE ON )

# use folders in IDEs for projects (e.g. lib sample app test)
set_property( GLOBAL PROPERTY USE_FOLDERS ON )

# Enable multithreading
find_package( Threads REQUIRED )

# set _WIN32_WINNT
if( WIN32 )
  # set _WIN32_WINT version global
  add_definitions( -D_WIN32_WINNT=0x0600 )
endif()

# enable parallel build for Visual Studio
if( MSVC )
  # add compile options
  add_compile_options( "/MP" )
  add_compile_options( "/EHsc" )
endif()

if( VVENC_ENABLE_TRACING )
  add_definitions( -DENABLE_TRACING=1 )
endif()

if( VVENC_ENABLE_X86_SIMD )
  if( ( UNIX OR MINGW ) AND VVENC_OPT_TARGET_ARCH )
    set( CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} -march=${VVENC_OPT_TARGET_ARCH} -mtune=${VVENC_OPT_TARGET_ARCH}" )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=${VVENC_OPT_TARGET_ARCH} -mtune=${VVENC_OPT_TARGET_ARCH}" )
  endif()
endif()

if( VVENC_ENABLE_LINK_TIME_OPT )
  set( CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON )
  set( CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO ON )
  set( CMAKE_INTERPROCEDURAL_OPTIMIZATION_MINSIZEREL ON )

  if( CMAKE_CXX_COMPILER_ID STREQUAL "GNU" )
    # enable parallel link-time optimization for GCC
    add_link_options( $<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>,$<CONFIG:MinSizeRel>>:-flto=auto> )
  endif()
endif()

if( VVCEncoderLib_ENABLE_ITT )
  if( MSVC )
    set( ITT_PATH "c:/Program Files (x86)/IntelSWTools/VTune Amplifier/" CACHE STRING "Path to the installation directory of Intel VTunes" )
  elseif( APPLE )
    message( WARNING "Not yet supported on Mac OS X" )
  elseif( UNIX OR MINGW )
    set( ITT_PATH "/opt/intel/vtune_amplifier" CACHE STRING "Path to the installation directory of Intel VTunes" )
  endif()

  if( EXISTS ${ITT_PATH} )
    set( LIB_NAME INTEL_ITT )
    add_library( ${LIB_NAME} STATIC IMPORTED GLOBAL )
    if( MSVC )
      set_target_properties( ${LIB_NAME} PROPERTIES IMPORTED_LOCATION        ${ITT_PATH}/lib64/libittnotify.lib )
    elseif( APPLE )
      # not supported
    elseif( UNIX OR MINGW )
      set_target_properties( ${LIB_NAME} PROPERTIES IMPORTED_LOCATION        ${ITT_PATH}/lib64/libittnotify.a )
      set_target_properties( ${LIB_NAME} PROPERTIES INTERFACE_LINK_LIBRARIES dl )
    endif()

    # set include directory. relative paths do not work.
    set_target_properties( ${LIB_NAME} PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${ITT_PATH}/include )
    set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DTRACE_ENABLE_ITT" )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTRACE_ENABLE_ITT" )
  else()
    message( WARNING "ITT_PATH ${ITT_PATH} not found, ignoring option VVCEncoderLib_ENABLE_ITT" )
  endif()
endif()

# set address sanitizer compiler arguments
if( VVENC_USE_ADDRESS_SANITIZER )
   # add compile options
   add_compile_options( "-fsanitize=address" )
   add_compile_options( "-fno-omit-frame-pointer" )
   add_compile_options( "-fsanitize=undefined" )
   add_compile_options( "-fsanitize=leak" )
   add_link_options( "-fsanitize=address" )
   add_link_options( "-fno-omit-frame-pointer" )
   add_link_options( "-fsanitize=undefined" )
   add_link_options( "-fsanitize=leak" )
endif()

# set thread sanitizer compiler arguments
if( VVENC_USE_THREAD_SANITIZER )
   # add compile options
   add_compile_options( "-fno-omit-frame-pointer" )
   add_compile_options( "-fsanitize=thread" )
   add_link_options( "-fno-omit-frame-pointer" )
   add_link_options( "-fsanitize=thread" )
endif()

# use ccache
find_program( CCACHE_FOUND ccache )
if( CCACHE_FOUND )
  message( STATUS "ccache found. using it." )
  set_property( GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache )
  set_property( GLOBAL PROPERTY RULE_LAUNCH_LINK ccache )
endif()


if( VVENC_ENABLE_THIRDPARTY_JSON )
  set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DVVENC_ENABLE_THIRDPARTY_JSON" )
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVVENC_ENABLE_THIRDPARTY_JSON" )
endif()

# handle rpath correctly
if( VVENC_ENABLE_INSTALL )
  if( CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT )
    set( CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/install" CACHE PATH "Standard install prefix" FORCE )
  endif()

  # use GNU install dirs
  include( GNUInstallDirs )
  
  if( BUILD_SHARED_LIBS AND NOT WIN32 )
    set( CMAKE_SKIP_INSTALL_RPATH OFF CACHE BOOL "skip rpath" )
    if( APPLE )
      set( RPATH_BASE @loader_path )
    elseif( UNIX )
      set( RPATH_BASE $ORIGIN )
    endif()
    
    file( RELATIVE_PATH RPATH_REL_DIR ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR} ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR} )
    
    set( CMAKE_INSTALL_RPATH ${RPATH_BASE} ${RPATH_BASE}/${RPATH_REL_DIR} )
    message( STATUS "CMAKE_INSTALL_RPATH=${CMAKE_INSTALL_RPATH}" )
  endif()  
endif()


add_subdirectory( "source/Lib/vvenc" )
add_subdirectory( "source/App/vvencapp" )
add_subdirectory( "source/App/vvencFFapp" )
add_subdirectory( "test/vvenclibtest" )
add_subdirectory( "test/vvencinterfacetest" )

include( cmake/modules/vvencTests.cmake )

if( VVENC_ENABLE_INSTALL )
  # include installer
  include( cmake/modules/vvencInstall.cmake )
endif()

