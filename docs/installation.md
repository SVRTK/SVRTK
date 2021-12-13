Installation
============

General Installation Instructions
---------------------------------

Follow these instructions to install a modified version of MIRTK with the SVRTK toolbox
compiled as a package. The installation instructions assume you are using a clean Ubuntu
installation, hence various required software are also downloaded, e.g.: Python, CMake, etc.
The software can also be installed on OSX.


Directory Setup
---------------

```bash
cd /path/to/software/installation/directory/of/your/choice
(i.e.: /Users/your-user-id/Software)

mkdir reconstruction-software
cd reconstruction-software
```

Install Required Software/Libraries: Ubuntu 16
----------------------------------------------

**Check all software up-to-date**
```bash
sudo apt-get update

sudo apt install git
sudo apt-get install cmake cmake-curses-gui
sudo apt-get install python
```

**Install libraries needed for MIRTK**
```bash
sudo apt-get install build-essential libtbb-dev libboost-all-dev libeigen3-dev zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget zram-config fltk1.3-dev libgl1-mesa-glx mesa-utils libglm-dev
```

Install Required Software/Libraries: Mac OS X
---------------------------------------------

**Install 'brew' package manager from https://brew.sh/:**
```bash
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

**Install libraries needed for MIRTK**
```bash
brew install git cmake boost tbb eigen fltk
```

Install SVRK and MIRTK
----------------------

**Download MIRTK (please use this version with additional SVRTK compilation options)**
```bash
git clone https://github.com/SVRTK/MIRTK.git
```

**Download SVRTK into /Packages folder of MIRTK**
```bash
git clone https://github.com/SVRTK/SVRTK.git MIRTK/Packages/SVRTK
```

Compile MIRTK with SVRTK
------------------------

```bash
cd MIRTK/
mkdir build
cd build/

cmake -D WITH_TBB="ON" -D MODULE_SVRTK="ON" ..
make -j
```

If this is all installed correctly, then:
- The main 'mirtk' executable file is located in /build/bin/ folder
- The executable files for all MIRTK and SVRTK functions are located in /build/lib/tools/ folder


Add Executable files to the Path
--------------------------------

**Edit path file using nano text editor:**
```bash
nano ~/.bash_profile
```

**or:**
```bash
nano ~/.bashrc
```

**Enter the bottom of the text editor:**
```bash
export PATH="$PATH:/Users/your-user-id/Software/reconstruction-software/MIRTK/build/bin"
export PATH="$PATH:/Users/your-user-id/Software/reconstruction-software/MIRTK/build/lib/tools"
```

**Activate:**
```bash
source ~/.bashrc
```

To test, try typing "reconstruct" in a terminal window. If it is working, the command help will show. You may need to reboot your computer

Troubleshooting
---------------

**If there are issues with missing libraries when compiling, use ccmake for more details:**
```bash
cd MIRTK/build
ccmake .. 	 # make sure WITH_TBB=ON
		 # make sure MODULE_SVRTK=ON
```

**If you cannot call the binaries (i.e.: typing "reconstruct" gives nothing), try editing .profile:**
```bash
nano ~/.profile instead of nano ~/.bash_profile
```

Troubleshooting: out of memory errors during runtime
----------------------------------------------------

In general SVR reconstruction works on machines with >= 32 RAM on Ubuntu and ~ 32G RAM on Mac OS X (due to different memory compression)

In case if you get "-9 error" (memory issue) - please run reconstructions with "-remote" option
It is slower - but it amount of RAM should not exceed 20 GB (for cardiac 4D or body/placenta large ROI)

**For Linux machines with < 32G RAM - it is recommended to increase swap memory based on the following instructions:**

**1. Allocate swap file with 32G (if RAM is less < 32G)**
```bash
sudo fallocate -l 32G /swapfile		# or 16-64G
```

**2. Update swap file permissions**
```bash
sudo chmod 600 /swapfile
```

**3. Set up a Linux swap area on the file**
```bash
sudo mkswap /swapfile
```

**4. Activate the swap file**
```bash
sudo swapon /swapfile
```

**5. Verify that the swap is active**
```bash
sudo swapon --show
```

Installation of SVRTK as a standalone package, if required
----------------------------------------------------------

```bash
git clone https://github.com/SVRTK/MIRTK.git
cd MIRTK/
mkdir build
cd build/
ccmake .. 	# select WITH_TBB ON
make -j
pwd # copy this path for use when building SVRTK in next steps. e.g., /Users/***/software/reconstruction-software/MIRTK/build

cd ../../
git clone https://github.com/SVRTK/SVRTK.git
cd SVRTK/
mkdir build
cd build/
ccmake ..	# paste path from earlier to DEPENDS_MIRTK_DIR
make -j

# the executable files for SVRTK functions are located in /build/lib/tools/ folder
```
