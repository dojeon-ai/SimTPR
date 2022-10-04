#!/bin/bash
games='Alien Amidar Assault Asterix BankHeist BattleZone Boxing Breakout ChopperCommand CrazyClimber DemonAttack Freeway Frostbite Gopher Hero Jamesbond Kangaroo Krull KungFuMaster MsPacman Pong PrivateEye Qbert RoadRunner Seaquest UpNDown'
runs='1 2'
ckpts='1 3 4 5 50'
files='action observation reward terminal'
export data_dir='./atari'


echo "Missing Files:"
for g in ${games[@]}; do
  for f in ${files[@]}; do
    for r in ${runs[@]}; do
        for c in ${ckpts[@]}; do
          if [ ! -f "${data_dir}/${g}/${f}_${r}_${c}.gz" ]; then
            echo "${data_dir}/${g}/${f}_${r}_${c}.gz"
          fi;
        done;
    done;
  done;
done;

for g in ${games[@]}; do
  mkdir -p "${data_dir}/${g}"
  for f in ${files[@]}; do
    for c in ${ckpts[@]}; do
      for r in ${runs[@]}; do
          if [ ! -f "${data_dir}/${g}/${f}_${r}_${c}.gz" ]; then
            gsutil cp "gs://atari-replay-datasets/dqn/${g}/${r}/replay_logs/\$store\$_${f}_ckpt.${c}.gz" "${data_dir}/${g}/${f}_${r}_${c}.gz"
          fi;
      done;
    done;
  done;
done;