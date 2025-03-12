import csv

csv_filename = 'speed_results.csv'
def write_speed_to_csv(track_id, class_name, direction, speed, csv_file=csv_filename):
    """
    Write the calculated speed information to a CSV file.
    """
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([track_id, class_name, direction, speed])