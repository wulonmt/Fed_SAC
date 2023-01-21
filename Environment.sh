#!/bin/sh
echo "Stating Environment"

SETTINGS=$PWD/settings.json
echo $SETTINGS
~/AirSim/Unreal/Environments/AirSimNH/LinuxNoEditor/AirSimNH.sh \
-windowed -ResX=1024 -ResY=576 -NoVSync \
-settings=$SETTINGS


exit 0
