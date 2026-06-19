import csv

with open("data/tracks_view001.csv", "r", newline="") as f1:
    rows1 = list(csv.reader(f1))

with open("data/tracks_view005.csv", "r", newline="") as f2:
    rows2 = list(csv.reader(f2))

with open("data/detections.csv", "w", newline="", encoding="utf-8") as out:
    writer = csv.writer(out)

    # write first file completely
    writer.writerows(rows1)

    # skip header of second file
    writer.writerows(rows2[1:])

print("detections.csv recreated successfully")