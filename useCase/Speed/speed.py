def calculate_speed(start_time, end_time, distance):
    """
    Calculate speed given start and end times and a known distance.
    """
    elapsed_time = end_time - start_time
    if elapsed_time <= 0:
        return 0
    return distance / elapsed_time