#!/bin/bash
games='Alien Amidar Assault Asterix BankHeist BattleZone Boxing Breakout ChopperCommand CrazyClimber DemonAttack Freeway Frostbite Gopher Hero Jamesbond Kangaroo Krull KungFuMaster MsPacman Pong PrivateEye Qbert RoadRunner Seaquest UpNDown'
ckpts='1 5 10 15 20 25 30 40 50'
files='action observation reward terminal'
export data_dir='/home/nas3_userK/hojoonlee/projects/video_rl/data/atari'


echo "Missing Files:"
for g in ${games[@]}; do
  for f in ${files[@]}; do
    for c in ${ckpts[@]}; do
      if [ ! -f "${data_dir}/${g}/${f}_${c}.gz" ]; then
        echo "${data_dir}/${g}/${f}_${c}.gz"
      fi;
    done;
  done;
done;

"""
# https://stackoverflow.com/a/226724
echo "Do you wish to download missing files?"
select yn in "Yes" "No"; do
    case $yn in
        Yes ) break;;
        No ) exit;;
    esac
done
"""

for g in ${games[@]}; do
  mkdir -p "${data_dir}/${g}"
  for f in ${files[@]}; do
    for c in ${ckpts[@]}; do
      if [ ! -f "${data_dir}/${g}/${f}_${c}.gz" ]; then
        gsutil cp "gs://atari-replay-datasets/dqn/${g,,}/1/replay_logs/\$store\$_${f}_ckpt.${c}.gz" "${data_dir}/${g}/${f}_${c}.gz"
      fi;
    done;
  done;
done;