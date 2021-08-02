Change the kernel.cu -> MAX_ITERATIONS param to a higher number to get higher definition results.
Compile with -DSHOW_ONLY_ITER=TRUE to only show iterations as image.

# Setup
## Windows
1. Install CUDA toolkit from [here](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64). Note: lib has been tested with CUDA 11.04
2. Go to 'C:\ProgramData\NVIDIA Corporation\CUDA Samples\v11.4'
3. Copy the 'common' folder into this repo
### Clion
#### Cmake
4. open the project in Clion as 'Cmake project'
5. compile the programm. Note that execution will fail.
6. from the 'C:\ProgramData\NVIDIA Corporation\CUDA Samples\v11.4\bin\win64\release\' folder copy the 'freeglut.dll' and 'glew64.dll' into the './cmake-build-debug' folder created by clion
7. Run the programm again and now it should execute
### Visual Studio 2019
4. From 'C:\ProgramData\NVIDIA Corporation\CUDA Samples\v11.4' copy the 'bin' folder into this repo
4. Open the properties of the Solution
5. 'Configuration Properties >> C/C++ >> General' in 'AdditionalIncludeDirectories' add ';common/inc' 
6. 'Configuration Properties >> Linker >> General' in 'Additional Library Directories' add ';common/lib' 
6. 'Configuration Properties >> Linker >> Input' in 'Additional Library Directories' add 'glew64.lib' 
8. 'Configuration Properties >> General' change 'Output Directory' to '$(ProjectDir)bin/win64/$(Configuration)/' 
## Linux
1. TODO Install ???
2. ...
3. Use CMake or Make
# Sources
Large parts of Code taken from https://www.informit.com/articles/article.aspx?p=2455391&seqNum=2