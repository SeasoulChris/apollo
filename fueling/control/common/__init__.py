def get_vehicle_type(data_folder):
    sub_folders = os.listdir(data_folder)
    vehicle_types = []
    for one_folder in sub_folders:
        vehicle_types.append(one_folder.rsplit('/'))
    return vehicle_types
