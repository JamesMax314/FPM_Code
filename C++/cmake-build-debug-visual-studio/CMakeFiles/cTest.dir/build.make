# CMAKE generated file: DO NOT EDIT!
# Generated by "NMake Makefiles" Generator, CMake Version 3.14

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

!IF "$(OS)" == "Windows_NT"
NULL=
!ELSE
NULL=nul
!ENDIF
SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"

# The command to remove a file.
RM = "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = D:\MainDrive\University\SummerPlacement\FPM\Code\NewBuild\cTestE

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = D:\MainDrive\University\SummerPlacement\FPM\Code\NewBuild\cTestE\cmake-build-debug-visual-studio

# Include any dependencies generated for this target.
include CMakeFiles\cTest.dir\depend.make

# Include the progress variables for this target.
include CMakeFiles\cTest.dir\progress.make

# Include the compile flags for this target's objects.
include CMakeFiles\cTest.dir\flags.make

CMakeFiles\cTest.dir\main.cpp.obj: CMakeFiles\cTest.dir\flags.make
CMakeFiles\cTest.dir\main.cpp.obj: ..\main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=D:\MainDrive\University\SummerPlacement\FPM\Code\NewBuild\cTestE\cmake-build-debug-visual-studio\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/cTest.dir/main.cpp.obj"
	C:\PROGRA~2\MIB055~1\2019\COMMUN~1\VC\Tools\MSVC\1422~1.279\bin\Hostx64\x64\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoCMakeFiles\cTest.dir\main.cpp.obj /FdCMakeFiles\cTest.dir\ /FS -c D:\MainDrive\University\SummerPlacement\FPM\Code\NewBuild\cTestE\main.cpp
<<

CMakeFiles\cTest.dir\main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cTest.dir/main.cpp.i"
	C:\PROGRA~2\MIB055~1\2019\COMMUN~1\VC\Tools\MSVC\1422~1.279\bin\Hostx64\x64\cl.exe > CMakeFiles\cTest.dir\main.cpp.i @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E D:\MainDrive\University\SummerPlacement\FPM\Code\NewBuild\cTestE\main.cpp
<<

CMakeFiles\cTest.dir\main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cTest.dir/main.cpp.s"
	C:\PROGRA~2\MIB055~1\2019\COMMUN~1\VC\Tools\MSVC\1422~1.279\bin\Hostx64\x64\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoNUL /FAs /FaCMakeFiles\cTest.dir\main.cpp.s /c D:\MainDrive\University\SummerPlacement\FPM\Code\NewBuild\cTestE\main.cpp
<<

# Object files for target cTest
cTest_OBJECTS = \
"CMakeFiles\cTest.dir\main.cpp.obj"

# External object files for target cTest
cTest_EXTERNAL_OBJECTS =

cTest.exe: CMakeFiles\cTest.dir\main.cpp.obj
cTest.exe: CMakeFiles\cTest.dir\build.make
cTest.exe: C:\Users\James\Anaconda3\Library\lib\fftw3.lib
cTest.exe: D:\Cpp_Libs\installs\x64\vc16\lib\opencv_dnn411d.lib
cTest.exe: D:\Cpp_Libs\installs\x64\vc16\lib\opencv_gapi411d.lib
cTest.exe: D:\Cpp_Libs\installs\x64\vc16\lib\opencv_highgui411d.lib
cTest.exe: D:\Cpp_Libs\installs\x64\vc16\lib\opencv_ml411d.lib
cTest.exe: D:\Cpp_Libs\installs\x64\vc16\lib\opencv_objdetect411d.lib
cTest.exe: D:\Cpp_Libs\installs\x64\vc16\lib\opencv_photo411d.lib
cTest.exe: D:\Cpp_Libs\installs\x64\vc16\lib\opencv_stitching411d.lib
cTest.exe: D:\Cpp_Libs\installs\x64\vc16\lib\opencv_video411d.lib
cTest.exe: D:\Cpp_Libs\installs\x64\vc16\lib\opencv_videoio411d.lib
cTest.exe: D:\Cpp_Libs\installs\x64\vc16\lib\opencv_imgcodecs411d.lib
cTest.exe: D:\Cpp_Libs\installs\x64\vc16\lib\opencv_calib3d411d.lib
cTest.exe: D:\Cpp_Libs\installs\x64\vc16\lib\opencv_features2d411d.lib
cTest.exe: D:\Cpp_Libs\installs\x64\vc16\lib\opencv_flann411d.lib
cTest.exe: D:\Cpp_Libs\installs\x64\vc16\lib\opencv_imgproc411d.lib
cTest.exe: D:\Cpp_Libs\installs\x64\vc16\lib\opencv_core411d.lib
cTest.exe: CMakeFiles\cTest.dir\objects1.rsp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=D:\MainDrive\University\SummerPlacement\FPM\Code\NewBuild\cTestE\cmake-build-debug-visual-studio\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable cTest.exe"
	"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" -E vs_link_exe --intdir=CMakeFiles\cTest.dir --rc=C:\PROGRA~2\WI3CF2~1\10\bin\100183~1.0\x64\rc.exe --mt=C:\PROGRA~2\WI3CF2~1\10\bin\100183~1.0\x64\mt.exe --manifests  -- C:\PROGRA~2\MIB055~1\2019\COMMUN~1\VC\Tools\MSVC\1422~1.279\bin\Hostx64\x64\link.exe /nologo @CMakeFiles\cTest.dir\objects1.rsp @<<
 /out:cTest.exe /implib:cTest.lib /pdb:D:\MainDrive\University\SummerPlacement\FPM\Code\NewBuild\cTestE\cmake-build-debug-visual-studio\cTest.pdb /version:0.0  /machine:x64 /debug /INCREMENTAL /subsystem:console C:\Users\James\Anaconda3\Library\lib\fftw3.lib D:\Cpp_Libs\installs\x64\vc16\lib\opencv_dnn411d.lib D:\Cpp_Libs\installs\x64\vc16\lib\opencv_gapi411d.lib D:\Cpp_Libs\installs\x64\vc16\lib\opencv_highgui411d.lib D:\Cpp_Libs\installs\x64\vc16\lib\opencv_ml411d.lib D:\Cpp_Libs\installs\x64\vc16\lib\opencv_objdetect411d.lib D:\Cpp_Libs\installs\x64\vc16\lib\opencv_photo411d.lib D:\Cpp_Libs\installs\x64\vc16\lib\opencv_stitching411d.lib D:\Cpp_Libs\installs\x64\vc16\lib\opencv_video411d.lib D:\Cpp_Libs\installs\x64\vc16\lib\opencv_videoio411d.lib D:\Cpp_Libs\installs\x64\vc16\lib\opencv_imgcodecs411d.lib D:\Cpp_Libs\installs\x64\vc16\lib\opencv_calib3d411d.lib D:\Cpp_Libs\installs\x64\vc16\lib\opencv_features2d411d.lib D:\Cpp_Libs\installs\x64\vc16\lib\opencv_flann411d.lib D:\Cpp_Libs\installs\x64\vc16\lib\opencv_imgproc411d.lib D:\Cpp_Libs\installs\x64\vc16\lib\opencv_core411d.lib kernel32.lib user32.lib gdi32.lib winspool.lib shell32.lib ole32.lib oleaut32.lib uuid.lib comdlg32.lib advapi32.lib 
<<

# Rule to build all files generated by this target.
CMakeFiles\cTest.dir\build: cTest.exe

.PHONY : CMakeFiles\cTest.dir\build

CMakeFiles\cTest.dir\clean:
	$(CMAKE_COMMAND) -P CMakeFiles\cTest.dir\cmake_clean.cmake
.PHONY : CMakeFiles\cTest.dir\clean

CMakeFiles\cTest.dir\depend:
	$(CMAKE_COMMAND) -E cmake_depends "NMake Makefiles" D:\MainDrive\University\SummerPlacement\FPM\Code\NewBuild\cTestE D:\MainDrive\University\SummerPlacement\FPM\Code\NewBuild\cTestE D:\MainDrive\University\SummerPlacement\FPM\Code\NewBuild\cTestE\cmake-build-debug-visual-studio D:\MainDrive\University\SummerPlacement\FPM\Code\NewBuild\cTestE\cmake-build-debug-visual-studio D:\MainDrive\University\SummerPlacement\FPM\Code\NewBuild\cTestE\cmake-build-debug-visual-studio\CMakeFiles\cTest.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles\cTest.dir\depend

