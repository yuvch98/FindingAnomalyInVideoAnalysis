# importing needed libraries
import math
import os
import random
from sklearn.model_selection import train_test_split
random.seed(42)
#exp This code mainly runs in colab, hence why the data comes from Google Drive
def get_data():
    Abuse = os.listdir("/content/drive/MyDrive/all files/Video Data/Anomaly-Videos-Part-1/Abuse")
    for i in range(len(Abuse)):
        Abuse[i] = "/content/drive/MyDrive/all files/Video Data/Anomaly-Videos-Part-1/Abuse/" + Abuse[i]
    Arrest = os.listdir("/content/drive/MyDrive/all files/Video Data/Anomaly-Videos-Part-1/Arrest")
    for i in range(len(Arrest)):
        Arrest[i] = "/content/drive/MyDrive/all files/Video Data/Anomaly-Videos-Part-1/Arrest/" + Arrest[i]
    Arson = os.listdir("/content/drive/MyDrive/all files/Video Data/Anomaly-Videos-Part-1/Arson")
    for i in range(len(Arson)):
        Arson[i] = "/content/drive/MyDrive/all files/Video Data/Anomaly-Videos-Part-1/Arson/" + Arson[i]
    Assault = os.listdir("/content/drive/MyDrive/all files/Video Data/Anomaly-Videos-Part-1/Assault")
    for i in range(len(Assault)):
        Assault[i] = "/content/drive/MyDrive/all files/Video Data/Anomaly-Videos-Part-1/Assault/" + Assault[i]
    Burglary = os.listdir("/content/drive/MyDrive/all files/Video Data/Anomaly-Videos-Part-2/Burglary")
    for i in range(len(Burglary)):
        Burglary[i] = "/content/drive/MyDrive/all files/Video Data/Anomaly-Videos-Part-2/Burglary/" + Burglary[i]
    Explosion = os.listdir("/content/drive/MyDrive/all files/Video Data/Anomaly-Videos-Part-2/Explosion")
    for i in range(len(Explosion)):
        Explosion[i] = "/content/drive/MyDrive/all files/Video Data/Anomaly-Videos-Part-2/Explosion/" + Explosion[i]
    Fighting = os.listdir("/content/drive/MyDrive/all files/Video Data/Anomaly-Videos-Part-2/Fighting")
    for i in range(len(Fighting)):
        Fighting[i] = "/content/drive/MyDrive/all files/Video Data/Anomaly-Videos-Part-2/Fighting/" + Fighting[i]
    RoadAccidents = os.listdir("/content/drive/MyDrive/all files/Video Data/Anomaly-Videos-Part-3/RoadAccidents")
    for i in range(len(RoadAccidents)):
        RoadAccidents[i] = "/content/drive/MyDrive/all files/Video Data/Anomaly-Videos-Part-3/RoadAccidents/" + \
                           RoadAccidents[i]
    Robbery = os.listdir("/content/drive/MyDrive/all files/Video Data/Anomaly-Videos-Part-3/Robbery/")
    for i in range(len(Robbery)):
        Robbery[i] = "/content/drive/MyDrive/all files/Video Data/Anomaly-Videos-Part-3/Robbery/" + Robbery[i]
    Shooting = os.listdir("/content/drive/MyDrive/all files/Video Data/Anomaly-Videos-Part-3/Shooting")
    for i in range(len(Shooting)):
        Shooting[i] = "/content/drive/MyDrive/all files/Video Data/Anomaly-Videos-Part-3/Shooting/" + Shooting[i]
    Shoplifting = os.listdir("/content/drive/MyDrive/all files/Video Data/Anomaly-Videos-Part-4/Shoplifting")
    for i in range(len(Shoplifting)):
        Shoplifting[i] = "/content/drive/MyDrive/all files/Video Data/Anomaly-Videos-Part-4/Shoplifting/" + Shoplifting[
            i]
    Stealing = os.listdir("/content/drive/MyDrive/all files/Video Data/Anomaly-Videos-Part-4/Stealing")
    for i in range(len(Stealing)):
        Stealing[i] = "/content/drive/MyDrive/all files/Video Data/Anomaly-Videos-Part-4/Stealing/" + Stealing[i]
    Vandalism = os.listdir("/content/drive/MyDrive/all files/Video Data/Anomaly-Videos-Part-4/Vandalism")
    for i in range(len(Vandalism)):
        Vandalism[i] = "/content/drive/MyDrive/all files/Video Data/Anomaly-Videos-Part-4/Vandalism/" + Vandalism[i]

    # Test normal
    test_Normal = os.listdir("/content/drive/MyDrive/all files/Video Data/Testing_Normal_Videos_Anomaly")
    for i in range(len(test_Normal)):
        test_Normal[i] = "/content/drive/MyDrive/all files/Video Data/Testing_Normal_Videos_Anomaly/" + test_Normal[i]
    # Train normal
    Train_Normal = [os.listdir("/content/drive/MyDrive/all files/Video Data/Normal_Videos_for_Event_Recognition"),
                    os.listdir(
                        "/content/drive/MyDrive/all files/Video Data/Training-Normal-Videos-Part-1(1)/Training-Normal-Videos-Part-1"),
                    os.listdir(
                        "/content/drive/MyDrive/all files/Video Data/Training-Normal-Videos-Part-2(1)/Training-Normal-Videos-Part-2")]
    for i in range(3):
        for j in range(len(Train_Normal[i])):
            if i == 0:
                Train_Normal[i][
                    j] = "/content/drive/MyDrive/all files/Video Data/Normal_Videos_for_Event_Recognition/" + \
                         Train_Normal[i][j]
            if i == 1:
                Train_Normal[i][
                    j] = "/content/drive/MyDrive/all files/Video Data/Training-Normal-Videos-Part-1(1)/Training-Normal-Videos-Part-1/" + \
                         Train_Normal[i][j]
            if i == 2:
                Train_Normal[i][
                    j] = "/content/drive/MyDrive/all files/Video Data/Training-Normal-Videos-Part-2(1)/Training-Normal-Videos-Part-2/" + \
                         Train_Normal[i][j]
    Train_Normal = Train_Normal[0] + Train_Normal[1] + Train_Normal[2]
    amount_of_videos_per_type = min(
        [len(Abuse), len(Arrest), len(Arson), len(Assault), len(Burglary), len(Explosion), len(Fighting),
         len(RoadAccidents), len(Robbery), len(Shooting), len(Shoplifting), len(Stealing), len(Vandalism)])
    train_Abuse = Abuse[:math.ceil(amount_of_videos_per_type * 0.8)]
    test_Abuse = Abuse[math.ceil(amount_of_videos_per_type * 0.8):amount_of_videos_per_type]
    train_Arrest = Arrest[:math.ceil(amount_of_videos_per_type * 0.8)]
    test_Arrest = Arrest[math.ceil(amount_of_videos_per_type * 0.8):amount_of_videos_per_type]
    train_Arson = Arson[:math.ceil(amount_of_videos_per_type * 0.8)]
    test_Arson = Arson[math.ceil(amount_of_videos_per_type * 0.8):amount_of_videos_per_type]
    train_Assault = Assault[:math.ceil(amount_of_videos_per_type * 0.8)]
    test_Assault = Assault[math.ceil(amount_of_videos_per_type * 0.8):amount_of_videos_per_type]
    train_Burglary = Burglary[:math.ceil(amount_of_videos_per_type * 0.8)]
    test_Burglary = Burglary[math.ceil(amount_of_videos_per_type * 0.8):amount_of_videos_per_type]
    train_Explosion = Explosion[:math.ceil(amount_of_videos_per_type * 0.8)]
    test_Explosion = Explosion[math.ceil(amount_of_videos_per_type * 0.8):amount_of_videos_per_type]
    train_Fighting = Fighting[:math.ceil(amount_of_videos_per_type * 0.8)]
    test_Fighting = Fighting[math.ceil(amount_of_videos_per_type * 0.8):amount_of_videos_per_type]
    train_RoadAccidents = RoadAccidents[:math.ceil(amount_of_videos_per_type * 0.8)]
    test_RoadAccidents = RoadAccidents[math.ceil(amount_of_videos_per_type * 0.8):amount_of_videos_per_type]
    train_Robbery = Robbery[:math.ceil(amount_of_videos_per_type * 0.8)]
    test_Robbery = Robbery[math.ceil(amount_of_videos_per_type * 0.8):amount_of_videos_per_type]
    train_Shooting = Shooting[: math.ceil(amount_of_videos_per_type * 0.8)]
    test_Shooting = Shooting[math.ceil(amount_of_videos_per_type * 0.8):amount_of_videos_per_type]
    train_Shoplifting = Shoplifting[: math.ceil(amount_of_videos_per_type * 0.8)]
    test_Shoplifting = Shoplifting[math.ceil(amount_of_videos_per_type * 0.8):amount_of_videos_per_type]
    train_Stealing = Stealing[: math.ceil(amount_of_videos_per_type * 0.8)]
    test_Stealing = Stealing[math.ceil(amount_of_videos_per_type * 0.80):amount_of_videos_per_type]
    train_Vandalism = Vandalism[:math.ceil(amount_of_videos_per_type * 0.80)]
    test_Vandalism = Vandalism[math.ceil(amount_of_videos_per_type * 0.80):amount_of_videos_per_type]

    total_train_anomaly_size = math.ceil(amount_of_videos_per_type * 0.80) * 13
    total_test_anomaly_size = (amount_of_videos_per_type * 13) - math.ceil(amount_of_videos_per_type * 0.80) * 13
    all_Normal = Train_Normal + test_Normal
    random.shuffle(all_Normal)
    all_anomaly = train_Abuse + train_Arrest + train_Arson + train_Assault + train_Burglary + train_Explosion + train_Fighting + train_RoadAccidents + train_Robbery + train_Shooting + train_Shoplifting + train_Stealing + train_Vandalism + test_Abuse + test_Arrest + test_Arson + test_Assault + test_Burglary + test_Explosion + test_Fighting + test_RoadAccidents + test_Robbery + test_Shooting + test_Shoplifting + test_Stealing + test_Vandalism
    random.shuffle(all_anomaly)
    print(f"{all_anomaly}\n , {len(all_anomaly)}")
    # Redistribute the videos to have 559 for training and 391 for testing
    new_train_Normal = all_Normal[:math.ceil(amount_of_videos_per_type * 0.8) * 13 + 100]
    new_test_Normal = all_Normal[math.ceil(amount_of_videos_per_type * 0.8) * 13:amount_of_videos_per_type * 13]
    new_train_Anomaly = all_anomaly[:total_train_anomaly_size]
    new_test_Anomaly = all_anomaly[total_train_anomaly_size:(total_train_anomaly_size + total_test_anomaly_size)]

    train_file_paths = new_train_Normal + new_train_Anomaly
    test_file_paths = new_test_Normal + new_test_Anomaly
    random.shuffle(train_file_paths)
    random.shuffle(test_file_paths)
    return train_file_paths, test_file_paths
def get_label_from_path(file_path):
  normal = ["Normal", "normal"]
  for word in normal:
    if word in file_path:
      return 0
  return 1
def train_test_dataset(train_file_paths, test_file_paths):
    test_labels = [get_label_from_path(path) for path in test_file_paths]
    train_labels = [get_label_from_path(path) for path in train_file_paths]
    x_train, x_val, y_train, y_val = train_test_split(train_file_paths, train_labels, test_size=0.2, random_state=42)
    return x_train, x_val, y_train, y_val, test_labels
