#!/usr/bin/zsh
for split in 1 2 4 8;do
  for random in '' '--random';do
    for pretrained in '' '--pretrained';do
      python3 train.py --split $split $random $pretrained --savefolder data/split${split}${random}${pretrained}
    done
  done
done