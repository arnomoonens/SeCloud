#!/usr/bin/env python
# -*- coding: utf8 -*-

import json
import csv
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("source", type=str, help="Path to the json datafile.")
parser.add_argument("target", type=str, help="Path to where the csv file will be written.")

args = parser.parse_args()

with open(args.source) as f:
    res = json.load(f)

with open(args.target, "w") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["cl", "textRaw"], extrasaction="ignore", quoting=csv.QUOTE_NONNUMERIC)
    writer.writeheader()
    for dp in res:
        if dp["cl"] != -1:  # Don't include test data
            writer.writerow(dp)
