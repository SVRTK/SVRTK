Docker
======

Compiled SVR toolbox is available via [DockerHub](https://hub.docker.com/repository/docker/fetalsvrtk/svrtk):

```bash

docker pull fetalsvrtk/svrtk

docker run -it --rm --mount type=bind,source=location_on_your_machine,target=/home/data fetalsvrtk/svrtk /bin/bash

cd /home/data
mirtk reconstructBody ../outputDSVR.nii.gz 6 ../stack1.nii.gz ../stack2.nii.gz ../stack3.nii.gz ../stack4.nii.gz ../stack5.nii.gz ../stack6.nii.gz -mask ../mask.nii.gz -template ../stack2.nii.gz -thickness 2.5 -default -remote -resolution 0.85

```

_Notes: In order to make sure that reconstruction is fast enough - please select a sufficient number of CPUs (e.g., > 8) and amount of RAM (e.g., > 20 GB) in the Desktop Docker settings. You can increase RAM by using virtual RAM on Window and or swap on Ubuntu._
