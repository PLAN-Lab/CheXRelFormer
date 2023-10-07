import csv

#convert the dcm path to jpg for MIMIC-JPG
new_csv = []
with open('cxr-record-list.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    data = [row for row in reader]

# Skip the header row
header = data[0]
data = data[1:]
for subject_id, study_id, dicom_id, path in data:
    jpg_path = path.replace('.dcm', '.jpg')
    new_csv.append([subject_id, study_id, dicom_id, jpg_path])
with open('cxr-record-all-jpg.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)
    writer.writerows(new_csv)